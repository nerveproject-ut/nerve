#!/usr/bin/env python3
"""
Convert Infineon Position2Go radar frames from a .rad TLV file into a
radardb recording that a ``RadarBackend`` can open directly.

The .rad format multiplexes DAVIS346 events and Infineon radar ADC frames
in a single binary TLV stream.  ``rad_polarities_to_hdf5.py`` already
extracts the DVS events; this companion script extracts the **radar**
frames and writes them in the radardb HDF5 layout that the DSP pipeline
expects.

Output structure::

    <output_dir>/
    ├── recording.xml
    ├── meta_data/
    │   └── scenario.xml
    └── captured_data/
        └── set000/
            ├── infineon_p2g.xml      # per-device metadata
            └── data.h5               # HDF5 with radar/dataset_1/{data,timestamps,...}

Usage::

    python rad_radar_to_radardb.py -i /path/to/session/radar_and_davis346_events.rad \\
                                   -o /path/to/session/infineon_radar

    # Then use a RadarBackend to process:
    from nerve.radar import get_backend
    Backend = get_backend()
    radar = Backend.from_recording("/path/to/session/infineon_radar")
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.dom.minidom import getDOMImplementation

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from rad_file_parser import RadarFileParser


HARDWARE_ID = "infineon_p2g"
HDF5_GROUP_PATH = "radar/dataset_1"

# Fields that InfineonConfiguration.__slots__ accepts.
# See the Infineon radar driver's configuration module for the full list.
# Only these keys may appear in extra.configuration; any others
# cause an InputException when the radardb connector loads the dataset.
_INFINEON_CONFIG_SLOTS = {
    "numSamples", "numChirpLoops", "chirpPeriod", "framePeriod", "numFrames",
    "iqSamples", "startFrequency", "stopFrequency", "frequencySlope",
    "rxMask", "txSequence",
    "horizontalTxAntennaSpacing", "verticalTxAntennaSpacing",
    "horizontalRxAntennaSpacing", "verticalRxAntennaSpacing",
    "rxSamplingFrequency",
    "numHorizontalVirtualAntennas", "horizontalVirtualAntennaSpacing",
    "numVerticalVirtualAntennas", "verticalVirtualAntennaSpacing",
    "virtualAntennaMask", "calibrationData",
}

# Default Position2Go specs (from InfineonSettings defaults).
# Keys must be a subset of _INFINEON_CONFIG_SLOTS.
P2G_DEFAULTS = {
    "startFrequency": 24.025e9,
    "stopFrequency": 24.225e9,
    "rxMask": 3,            # both RX antennas
    "txSequence": [1],      # single TX
    "numSamples": None,     # filled from .rad header
    "numChirpLoops": None,  # filled from .rad header
    "numFrames": None,      # filled from actual frame count
    "iqSamples": None,      # filled from .rad header
    "framePeriod": 0.1,
    "rxSamplingFrequency": 213675.0,
    "horizontalTxAntennaSpacing": 0.0,
    "verticalTxAntennaSpacing": 0.0,
    "horizontalRxAntennaSpacing": 0.5,
    "verticalRxAntennaSpacing": 0.0,
    "numHorizontalVirtualAntennas": 2,
    "horizontalVirtualAntennaSpacing": 0.5,
    "numVerticalVirtualAntennas": 1,
    "verticalVirtualAntennaSpacing": 0.0,
}


def _build_infineon_config(frame_info, num_frames, frame_period=None):
    """Build a configuration dict matching InfineonConfiguration slots."""
    cfg = dict(P2G_DEFAULTS)

    cfg["numSamples"] = frame_info.num_samples_per_chirp
    cfg["numChirpLoops"] = frame_info.num_chirps_per_frame
    cfg["numFrames"] = num_frames
    cfg["iqSamples"] = frame_info.with_complex_samples

    num_rx = frame_info.num_rx_antennas
    cfg["rxMask"] = (1 << num_rx) - 1
    cfg["numHorizontalVirtualAntennas"] = num_rx
    cfg["txSequence"] = list(range(1, frame_info.num_tx_antennas + 1))

    bw = cfg["stopFrequency"] - cfg["startFrequency"]
    # Rough chirp period estimate from sampling frequency and num samples
    up_chirp_time = cfg["numSamples"] / cfg["rxSamplingFrequency"]
    cfg["chirpPeriod"] = up_chirp_time + 200e-6  # up + down(100us) + idle(100us)
    cfg["frequencySlope"] = bw / up_chirp_time

    if frame_period is not None:
        cfg["framePeriod"] = frame_period

    cfg["virtualAntennaMask"] = np.ones(
        (cfg["numVerticalVirtualAntennas"], cfg["numHorizontalVirtualAntennas"]),
        dtype=np.uint8,
    )
    cfg["calibrationData"] = np.ones(
        (frame_info.num_tx_antennas, num_rx), dtype=np.complex64
    )

    # Only keep keys that InfineonConfiguration accepts
    return {k: v for k, v in cfg.items() if k in _INFINEON_CONFIG_SLOTS}


def _reshape_td_matrix(td_matrix, frame_info):
    """Reshape a RadarFrame.td_matrix into the 5-D int16 radardb layout.

    The expected InfineonFrame format is:
        (chirp_loops, chirps_per_sequence, rx_antennas, samples, 1_or_2)
    where the last dimension is 1 (real) or 2 (I/Q interleaved as int16).

    The .rad parser already gives us complex128 shaped as:
        (chirps, [tx, [rx,]] samples)
    """
    ntx = frame_info.num_tx_antennas
    nrx = frame_info.num_rx_antennas
    nchirps = frame_info.num_chirps_per_frame
    nsamples = frame_info.num_samples_per_chirp
    is_complex = frame_info.with_complex_samples

    if td_matrix.ndim == 2:
        # (chirps, samples) -- SISO
        data = td_matrix.reshape(nchirps, 1, 1, nsamples)
    elif td_matrix.ndim == 4:
        # (chirps, tx, rx, samples)
        data = td_matrix
    else:
        raise ValueError(f"Unexpected td_matrix ndim={td_matrix.ndim}")

    # Ensure shape is (loops, tx_seq, rx, samples)
    data = data.reshape(nchirps, ntx, nrx, nsamples)

    if is_complex:
        real_part = data.real.astype(np.int16)
        imag_part = data.imag.astype(np.int16)
        out = np.stack([real_part, imag_part], axis=-1)  # (..., 2)
    else:
        out = data.real.astype(np.int16)[..., np.newaxis]  # (..., 1)

    return out


def _write_hdf5(output_dir, radar_frames, frame_info, config):
    """Write radar/dataset_1 group into data.h5."""
    capture_dir = output_dir / "captured_data" / "set000"
    capture_dir.mkdir(parents=True, exist_ok=True)
    h5_path = capture_dir / "data.h5"

    first_frame = _reshape_td_matrix(radar_frames[0].td_matrix, frame_info)
    frame_shape = first_frame.shape
    n = len(radar_frames)

    with h5py.File(str(h5_path), "w") as f:
        grp = f.create_group(HDF5_GROUP_PATH)

        ds_data = grp.create_dataset(
            "data",
            shape=(n,) + frame_shape,
            dtype="int16",
            chunks=(1,) + frame_shape,
            compression="gzip",
            compression_opts=1,
        )
        ds_ts = grp.create_dataset(
            "timestamps", shape=(n,), dtype="f8",
        )
        ds_seq = grp.create_dataset(
            "sequencenumbers", shape=(n,), dtype="u8",
        )
        ds_valid = grp.create_dataset(
            "validflags", shape=(n,), dtype="u1",
        )

        for i, rf in enumerate(radar_frames):
            ds_data[i] = _reshape_td_matrix(rf.td_matrix, frame_info)
            ds_ts[i] = float(rf.timestamps[0])
            ds_seq[i] = i
            ds_valid[i] = 1

        # Attach config as flattened group attributes (mirrors DatabaseRecorder)
        for key, val in config.items():
            attr_name = f"extra.configuration.{key}"
            if isinstance(val, np.ndarray):
                grp.attrs[attr_name] = val
            elif isinstance(val, list):
                grp.attrs[attr_name] = str(val)
            elif isinstance(val, bool):
                grp.attrs[attr_name] = str(val).lower()
            else:
                grp.attrs[attr_name] = val
        grp.attrs["hardwareId"] = HARDWARE_ID

    return h5_path, (n,) + frame_shape


def _add_xml_param(doc, parent, name, value, type_str):
    """Add <param name="..." value="..." type="..."/> to parent."""
    el = doc.createElement("param")
    el.setAttribute("name", name)
    el.setAttribute("value", str(value))
    el.setAttribute("type", type_str)
    parent.appendChild(el)


def _type_str(val):
    """Infer radardb XML type string from a Python value."""
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "string"
    if isinstance(val, np.ndarray):
        return f"array({val.dtype}, {', '.join(str(s) for s in val.shape)})"
    if isinstance(val, list):
        return f"list({type(val[0]).__name__})" if val else "list(int)"
    return "string"


def _write_device_xml(output_dir, config, h5_shape, start_time_str):
    """Write captured_data/set000/infineon_p2g.xml."""
    capture_dir = output_dir / "captured_data" / "set000"

    impl = getDOMImplementation()
    doc = impl.createDocument(None, "data_set", None)
    root = doc.documentElement
    root.setAttribute("version", "1")
    root.setAttribute("hardware_id", HARDWARE_ID)

    # <version> (minimal)
    ver_node = doc.createElement("version")
    root.appendChild(ver_node)

    # <config><extra><configuration>
    config_node = doc.createElement("config")
    root.appendChild(config_node)

    extra_node = doc.createElement("extra")
    config_node.appendChild(extra_node)

    cfg_node = doc.createElement("configuration")
    extra_node.appendChild(cfg_node)

    for key, val in config.items():
        if isinstance(val, np.ndarray):
            _add_xml_param(doc, cfg_node, key, repr(val), _type_str(val))
        elif isinstance(val, bool):
            _add_xml_param(doc, cfg_node, key, str(val).lower(), "bool")
        else:
            _add_xml_param(doc, cfg_node, key, val, _type_str(val))

    # <data>
    data_node = doc.createElement("data")
    data_node.setAttribute("kind", "radar")
    data_node.setAttribute("data_file", "data.h5")
    data_node.setAttribute("start_time", start_time_str)
    root.appendChild(data_node)

    container = doc.createElement("container")
    container.setAttribute("format", "hdf5")
    container.setAttribute("shape", str(h5_shape))
    container.setAttribute("dtype", "int16")
    container.setAttribute("path", HDF5_GROUP_PATH)
    data_node.appendChild(container)

    xml_path = capture_dir / f"{HARDWARE_ID}.xml"
    with open(xml_path, "w", encoding="utf-8") as f:
        doc.writexml(f, addindent="    ", newl="\n", encoding="utf-8")

    return xml_path


def _write_recording_xml(output_dir, start_time_str, end_time_str):
    """Write top-level recording.xml."""
    impl = getDOMImplementation()
    doc = impl.createDocument(None, "recording", None)
    root = doc.documentElement
    root.setAttribute("version", "1")
    root.setAttribute("xmlns:xi", "http://www.w3.org/2001/XInclude")

    # <scenarios>
    scenarios = doc.createElement("scenarios")
    root.appendChild(scenarios)
    inc_sc = doc.createElement("xi:include")
    inc_sc.setAttribute("href", "meta_data/scenario.xml")
    scenarios.appendChild(inc_sc)

    # <captured_data>
    captured = doc.createElement("captured_data")
    root.appendChild(captured)

    capture = doc.createElement("capture")
    capture.setAttribute("start_time", start_time_str)
    capture.setAttribute("end_time", end_time_str)
    captured.appendChild(capture)

    inc_dev = doc.createElement("xi:include")
    inc_dev.setAttribute("href", f"captured_data/set000/{HARDWARE_ID}.xml")
    capture.appendChild(inc_dev)

    xml_path = output_dir / "recording.xml"
    with open(xml_path, "w", encoding="utf-8") as f:
        doc.writexml(f, addindent="    ", newl="\n", encoding="utf-8")

    # Minimal scenario.xml
    meta_dir = output_dir / "meta_data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sc_doc = impl.createDocument(None, "scenario", None)
    sc_path = meta_dir / "scenario.xml"
    with open(sc_path, "w", encoding="utf-8") as f:
        sc_doc.writexml(f, addindent="    ", newl="\n", encoding="utf-8")


def convert(input_path, output_dir, frame_period=None, verbose=True):
    """Convert a .rad file's Infineon radar data to radardb format.

    Parameters
    ----------
    input_path : str or Path
        Path to the ``.rad`` TLV file.
    output_dir : str or Path
        Destination directory for the radardb recording.
    frame_period : float or None
        Override for frame period (seconds).  If *None*, estimated from
        timestamps in the file.
    verbose : bool
        Print progress information.

    Returns
    -------
    output_dir : Path
        The directory containing ``recording.xml``.
    num_frames : int
        Number of radar frames written.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if verbose:
        print(f"Parsing {input_path} ...")

    parser = RadarFileParser(str(input_path), read_radar=True, read_dvs=False)
    radar_frames = parser.radar_frames
    frame_info = parser._frame_info

    if len(radar_frames) == 0:
        print("No radar frames found in the .rad file.")
        return output_dir, 0

    if verbose:
        print(f"  Found {len(radar_frames)} radar frames")
        print(f"  TX={frame_info.num_tx_antennas}  RX={frame_info.num_rx_antennas}  "
              f"chirps={frame_info.num_chirps_per_frame}  samples={frame_info.num_samples_per_chirp}  "
              f"complex={frame_info.with_complex_samples}")

    # Estimate frame period from timestamps if not provided
    if frame_period is None and len(radar_frames) >= 2:
        t0 = float(radar_frames[0].timestamps[0])
        t1 = float(radar_frames[-1].timestamps[0])
        frame_period = (t1 - t0) / (len(radar_frames) - 1)
        if verbose:
            print(f"  Estimated frame period: {frame_period*1000:.1f} ms")

    config = _build_infineon_config(frame_info, len(radar_frames), frame_period)

    # Timestamps for XML
    t_start = float(radar_frames[0].timestamps[0])
    t_end = float(radar_frames[-1].timestamps[0])
    start_dt = datetime.fromtimestamp(t_start, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(t_end, tz=timezone.utc)
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    if verbose:
        print(f"Writing HDF5 to {output_dir} ...")

    h5_path, h5_shape = _write_hdf5(output_dir, radar_frames, frame_info, config)

    if verbose:
        print(f"  HDF5 data shape: {h5_shape}")
        print(f"Writing XML metadata ...")

    _write_device_xml(output_dir, config, h5_shape, start_str)
    _write_recording_xml(output_dir, start_str, end_str)

    if verbose:
        print(f"Done. radardb recording written to: {output_dir}")
        print(f"  {len(radar_frames)} frames, "
              f"covering {t_end - t_start:.1f}s")

    return output_dir, len(radar_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Infineon Position2Go radar data from .rad "
                    "to radardb format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the .rad TLV file",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for the radardb recording",
    )
    parser.add_argument(
        "--frame-period", type=float, default=None,
        help="Override frame period in seconds (estimated from timestamps "
             "if not provided)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_dir=args.output,
        frame_period=args.frame_period,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
