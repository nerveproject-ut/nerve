import json
import sys
import os

from nerve.extraction.utils.cameraParams import *

if __name__ == "__main__":
    
    
    file = "label_extraction/data_mapping/new_rgb_to_prophesee.json"
    """
    # INTEL RGB CAMERA:
    intel_intrinsics = IntrinsicParams(fx=905.030639648, fy=905.665710449, cx=653.86416574, cy=349.516906738)
    intel_distortions = StandardDistortionParams(k1=0.1414, k2=0.4705, k3=0.4217, p1=0.0017, p2=0.002)
    intel_cam = CameraParams(1280, 720, intel_intrinsics, intel_distortions, "rgb")


    # PROPHESEE DVS:
    proph_intrinsics = IntrinsicParams(fx=1.33039136e+03, fy=1.33039136e+03, cx=6.26509528e+02, cy=3.88429082e+02)
    proph_distortions = StandardDistortionParams(k1=-1.33959034e-01, k2=2.75107512e-01, p1=8.35428664e-04, p2=-5.59913906e-04, k3=-2.98488600e+01, k4=1.93668686e-01, k5=9.13103072e-01, k6=-3.47121907e+01)
    prophesee_cam = CameraParams(1280, 720, proph_intrinsics, proph_distortions, "prophesee")


    #mapping RGB -> Prophesee
    rotation_rgb_to_prophesee = np.array([  0.9991,	    0.0135,	    -0.0390, 
                                            -0.0152,	0.9989,	    -0.0437, 
                                            0.0383,	    0.0442,	    0.9983], dtype=np.float32).reshape((3,3))
    trans_rgb_to_prophesee = np.array([0.0212198, 0.0290229, 0.0130208], dtype=np.float32)
    intel_to_proph_extrinsics = ExtrinsicParams(rotation_rgb_to_prophesee, trans_rgb_to_prophesee)
    rgb_to_prophesee = Mapping(intel_cam,prophesee_cam, intel_to_proph_extrinsics)


    # Serializing json
    json_object = json.dumps(rgb_to_prophesee.toDict(), indent=4)
    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)
    """
    map = Mapping.from_file(file)
    

    """
    file = "data-mapping/L515_rgb_to_davis.json"
    # INTEL RGB CAMERA:
    intel_intrinsics = IntrinsicParams(fx=905.030639648, fy=905.665710449, cx=653.86416574, cy=349.516906738)
    intel_distortions = DistortionParams(k1=0.1414, k2=0.4705, k3=0.4217, p1=0.0017, p2=0.002)
    intel_cam = CameraParams(1280, 720, intel_intrinsics, intel_distortions, "rgb")


    # DAVIS DVS:
    davis_intrinsics = IntrinsicParams(fx=465.4087, fy=465.2258, cx=181.0787, cy=129.7596, skew=-0.0751)
    davis_distortions = DistortionParams(k1=-0.0551, k2=0.0168, k3=1.1436, p1=-1.646e-4, p2=0.0023)
    davis_cam = CameraParams(346, 260, davis_intrinsics, davis_distortions, "davis")


    #mapping RGB -> DAVIS
    rotation_rgb_to_davis = np.array([0.9994,   -0.0057,    -0.0338, 
                                      0.0039,   0.9985,	    -0.0545, 
                                      0.0340,   0.0543,	    0.9979], dtype=np.float32).reshape((3,3))
    
    trans_rgb_to_davis = np.array([-0.0282570, 0.0288121, 0.0410742], dtype=np.float32)
    intel_to_davis_extrinsics = ExtrinsicParams(rotation_rgb_to_davis, trans_rgb_to_davis)
    rgb_to_davis = Mapping(intel_cam, davis_cam, intel_to_davis_extrinsics)


    # Serializing json
    json_object = json.dumps(rgb_to_davis.toDict(), indent=4)
    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)
    """



    
    