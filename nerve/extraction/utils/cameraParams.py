import numpy as np
import json
import typing



class IntrinsicParams:
    def __init__(self, fx:float, fy:float, cx:float, cy:float, skew:float=0) -> None:
        """
        Parameters
        ----------
        fx, fy:
            The focal distances
        cx, cy : 
            Center of projection in pixels            
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.skew = skew
        self.matrix = np.array([fx, skew , cx, 0, fy, cy, 0, 0, 1], dtype=np.float32).reshape((3,3))

    @classmethod
    def from_dict(cls, dict:dict):
        return cls(dict['fx'], dict['fy'], dict['cx'], dict['cy'], dict['skew'])


    def toDict(self):
        return {'fx':self.fx, 'fy':self.fy, 'cx':self.cx, 'cy':self.cy, 'skew':self.skew}

    def __str__(self):
        return "Intrinsic params:\n{}\n".format(self.matrix)

class StandardDistortionParams:
    def __init__(self, distortion_vector:np.ndarray) -> None:
        """
        Parameters
        ----------
        [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy]

        where:
            + k1 -> k6      are radial coeffs
            + p1, p2        are tangential coeffs
            + s1 -> s4      are coeffs for the thin prism model
            + taux, tauy    are coeffs for the tilted model
        
        You can provide the firsts 4, 5, 8, 12 elements, or all of them (14)

        k1 [,k2 [, ..kn]]:
            The radial coefficients of the lens. Set all of them to 0 if you don't have distorsion.
            As note, k1 is typically negative for barrel distortions, while it is typically positive for pincushion distorsions.
        p1, p2 : 
            Tangential distortion of the lens (due to lens not perfectly parallel to image plane).
            Set to 0 if alignment is perfect.
        """
        assert distortion_vector.ndim == 1
        numel = distortion_vector.size
        assert numel in [4, 5, 8, 12, 14], "The given vector is incompatible, having {} elements.".format(numel)

        self.dist_vector = distortion_vector.astype(np.float32)

    @classmethod
    def from_dict(cls, dict:dict):
        try:
            dist_vector = np.asarray(dict['dist_vector'])
        except:
            # for backcompatibility
            dist_vector = np.array([dict['k1'], dict['k2'], dict['p1'], dict['p2'], dict['k3']])
        return cls(dist_vector)

    def toDict(self):
        return {'dist_vector':self.dist_vector.tolist()}

    def __str__(self):
        names = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4', 'taux', 'tauy']
        res = "Distorsion params:\n"
        for idx, val in enumerate(self.dist_vector):
            res += "\t{}:\t{}\n".format(names[idx], val)
        return res

class FishEyeDistortionParams:
    def __init__(self, k1:float, k2:float, k3:float, k4:float) -> None:
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        
        self.dist_vector = np.array([k1, k2, k3, k4], dtype=np.float32)
        """ Vector saved in format (k1, k2, k3, k4) to be compatible with openCV """

    @classmethod
    def from_dict(cls, dict:dict):
        return cls(dict['k1'], dict['k2'], dict['k3'], dict['k4'])

    def toDict(self):
        return {'k1':self.k1, 'k2':self.k2, 'k3':self.k3, 'k4':self.k4}

    def __str__(self):
        return "Distorsion params ( k1 | k2 | k3 | k4 ):\t{}\n".format(self.dist_vector)

class Device:
    def __init__(self, name:str) -> None:
        # eventually add other common params
        self.name = name
        pass

    def toDict(self):
        return {'name':self.name}
    
    def __str__(self):
        res = "\n-- Device {} :--\n".format(self.name)
        return res

    @classmethod
    def from_dict(cls, dict:dict):
        return cls(dict['name'])

class CameraParams(Device):
    def __init__(self, width:int, height:int,  intrinsics: IntrinsicParams, distortions:typing.Union[StandardDistortionParams, FishEyeDistortionParams], name:str) -> None:
        super().__init__(name)
        self.width=width
        self.height=height
        self.intrinsics = intrinsics
        self.distortions = distortions
        self.is_fisheye = isinstance(distortions, FishEyeDistortionParams)

    @classmethod
    def from_dict(cls, dict:dict):
        is_fisheye = dict['fisheye'] if 'fisheye' in dict.keys() else False
        dist = FishEyeDistortionParams.from_dict(dict['distortions']) if is_fisheye else StandardDistortionParams.from_dict(dict['distortions'])
        return cls(dict['width'], dict['height'], IntrinsicParams.from_dict(dict['intrinsics']), dist, dict['name'])

    @classmethod
    def from_file(cls, json_file_path:str):
         # Opening JSON file
        with open(json_file_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)

        p = CameraParams.from_dict(json_object)
        print("Loaded data from {}:{}".format(json_file_path, str(p)))
        return p

    def toDict(self):
        dict = super().toDict()
        specific_dict = {'width':self.width, 'height':self.height, 'intrinsics':self.intrinsics.toDict(), 'fisheye':self.is_fisheye , 'distortions':self.distortions.toDict()}
        dict.update(specific_dict)
        return dict

    def __str__(self):
        res = "\n-- Camera {} :--\n".format(self.name)
        res += "W x H: ({}, {})\n".format(self.width, self.height)
        res += str(self.intrinsics)
        res += "\nDistortion type: "
        if self.is_fisheye:
            res += "Fisheye\n"
        else:
            res += "Standard\n"
        res += str(self.distortions)
        return res
    

class ExtrinsicParams:
    def __init__(self, rot_matrix:np.ndarray, trans_vector:np.ndarray) -> None:

        assert rot_matrix.shape == (3,3)
        assert trans_vector.shape == (3,)

        self.rot_matrix = rot_matrix
        self.trans_vector = trans_vector

        rot_trans = np.hstack((rot_matrix, trans_vector.reshape(3,1)))
        self.matrix = np.vstack((rot_trans, np.array([0,0,0,1],dtype=np.float32)))

    @classmethod
    def from_dict(cls, dict:dict):
        return cls(np.asarray(dict['rot']), np.asarray(dict['trans']))

    def toDict(self):
        return {'rot':self.rot_matrix.tolist(), 'trans':self.trans_vector.tolist()}

    def __str__(self):
        return "\n-- Extrinsic params :--\n{}\n".format(self.matrix)
    

class Mapping:
    def __init__(self, src_dev:Device, dst_dev:Device, extr_A_to_B:ExtrinsicParams) -> None:
        self.src = src_dev
        self.dst = dst_dev
        self.extrinsics = extr_A_to_B

    @classmethod
    def from_dict(cls, dict:dict):
        try:
            src = CameraParams.from_dict(dict['src'])
        except:
            src = Device.from_dict(dict['src'])

        try:
            dst = CameraParams.from_dict(dict['dst'])
        except:
            dst = Device.from_dict(dict['dst'])

        return cls(src, dst, ExtrinsicParams.from_dict(dict['src_to_dst']))

    @classmethod
    def from_file(cls, json_file_path:str, verbose=False):
        # Opening JSON file
        with open(json_file_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)

        map = Mapping.from_dict(json_object)
        if verbose:
            print("Loaded mapping from {}:{}".format(json_file_path, str(map)))
        return map

    def toDict(self):
        return {'src':self.src.toDict(), 'dst':self.dst.toDict(), 'src_to_dst': self.extrinsics.toDict()}

    def __str__(self):
        type_src = "camera" if isinstance(self.src, CameraParams) else "device"
        type_dst = "camera" if isinstance(self.dst, CameraParams) else "device"

        res = "\nMapping from {} {} to {} {}:\n".format(type_src, self.src.name, type_dst, self.dst.name)
        res += "-----------------------------------------"
        res += "\nSource:"
        res += str(self.src)
        res += "\nDestination:"
        res += str(self.dst)
        res += str(self.extrinsics)
        res += "-----------------------------------------\n\n"
        return res
    
