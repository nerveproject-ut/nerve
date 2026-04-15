import struct
import sys
import os
import datetime


def getDataArrivalTime(sync_file_path: str, syncProtocolVersion: int = 0x03):
    assert sync_file_path.endswith(".sync")
    assert os.path.isfile(sync_file_path)

    with open(sync_file_path, "rb") as f:
        assert syncProtocolVersion == 0x03, "Error: undefined behaviour for sync protocol with version {}".format(syncProtocolVersion)
        f.read(0x13)
        buf = f.read(8)
        return struct.unpack_from("<q", buf)[0] / 1e6


if __name__ == "__main__":
    if len(sys.argv) == 2:
        time = getDataArrivalTime(sys.argv[1])
        print(time, " => ", datetime.datetime.fromtimestamp(time).strftime('%H:%M:%S.%f'))
    else:
        time1 = getDataArrivalTime(sys.argv[1])
        time2 = getDataArrivalTime(sys.argv[2])
        print("1): ", time1, " => ", datetime.datetime.fromtimestamp(time1).strftime('%H:%M:%S.%f'))
        print("2): ", time2, " => ", datetime.datetime.fromtimestamp(time2).strftime('%H:%M:%S.%f'))
        diff = (time2 - time1)*1e3
        print("Difference: {:.2f} mS.".format(diff))