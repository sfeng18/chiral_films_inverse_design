# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:03:46 2022

@author: DELL
"""

import os
import numpy as np
import ujson
import zlib
import msgpack
from chardet import detect

nums = [str(i) for i in range(10)]
Color_Dict = {'蓝2': 'blue2', '蓝6': 'blue6', '蓝71': 'blue71','刚果红':'congo red','绿':'green','橘红':'orange','紫':'purple','红':'red','台盼蓝':'tai','黄':'yellow'}
YName_Dict = {'G': 'G', 'CD': 'CD', 'ABSORBANCE': 'ABS'}

if msgpack.version[0] >= 1:
    msg_load_kwarg = {'strict_map_key': False}
else:
    msg_load_kwarg = {'encoding': 'utf-8'}


def save_as_json(filename, something):
    """Save something as json into file"""  #save something as json to a file
    with open(filename, 'wt') as f:
        ujson.dump(something, f)  #The fastest module to read json is ujson, and json.dump() writes data to the file in the data type of json


def load_from_json(filename):
    """Load something from json file"""  #load something from a json file
    if not os.path.exists(filename):  #Command exists: filename string as parameter, it will return True if the file exists, otherwise it will return False.
        raise AssertionError('File not found: %s' % filename)
    with open(filename, 'rt') as f:
        something = ujson.load(f)  #json.load() reads data from a json file
    return something


def msg_dumps_and_zip(something):
    return zlib.compress(msgpack.dumps(something, use_bin_type=True))


def msg_unzip_and_loads(something):
    return msgpack.loads(zlib.decompress(something), use_list=False, **msg_load_kwarg)


def load_from_msgzip(filename):
    """Load something from msgzip file"""
    if not os.path.exists(filename):
        raise AssertionError('File not found: %s' % filename)
    with open(filename, 'rb') as f:
        something = f.read()
    return msg_unzip_and_loads(something)


def save_as_msgzip(filename, something):
    """Save something as msgzip into file"""
    with open(filename, 'wb') as f:
        f.write(msg_dumps_and_zip(something))


def read_file(filename):
    """Read .json/msgz file"""
    assert os.path.isfile(filename), 'ERROR: Fail to find file %s\n' % filename
    temp = os.path.splitext(os.path.split(filename)[1])
    ext = temp[1][1:]
    if ext == 'json':
        return load_from_json(filename)
    elif ext == 'msgz':
        return load_from_msgzip(filename)
    else:
        raise AssertionError('Unrecognized file format!')


def save_file(filename, something):
    """Save .json/msgz file"""
    temp = os.path.splitext(os.path.split(filename)[1])
    ext = temp[1][1:]
    if ext == 'json':
        return save_as_json(filename, something)
    elif ext == 'msgz':
        return save_as_msgzip(filename, something)
    else:
        raise AssertionError('Unrecognized file format!')


def read_txt_lines(FileName, SPLIT=False, IncludeComment=False):
    """Read lines from txt file, skipping blank and comment lines. Return a list."""  #Read lines from a txt file, skipping blank and commented lines. Returns a list.
    with open(FileName, 'rb') as f:
        cur_encoding = detect(f.read(10000))['encoding']
    with open(FileName, 'rt', encoding=cur_encoding) as f:
        lines = f.readlines()
    TreatMethod = str.split if SPLIT else str.strip
    return list(TreatMethod(l) for l in lines if l.strip() and (IncludeComment or l.strip()[0] != '#'))


class SpectrumData(object):
    """Informations of Spectrum Data"""  #Spectral data information

    def __init__(self, ID=None, NAME='', COLOR='', ANGLE=0, THICK=0, STRAIN=0, GRAY=0, POLAR=''):
        self.ID = ID
        self.Name = NAME
        self.Color = COLOR  # string
        self.Angle = ANGLE  # Floating-point numbers are convenient to use directly as parameters for machine learning
        self.Thickness = THICK  # floating point number
        self.Strain = STRAIN  # floating point number
        self.Grayscale = GRAY  # floating point number
        self.PolarType = POLAR  # string
        self.X = []
        self.G = []
        self.CD = []
        self.ABS = []

    def __eq__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return (self.Name == other.Name)

    def __gt__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        if self.Color != other.Color:
            return (self.Color > other.Color)
        elif self.PolarType != other.PolarType:
            return (self.PolarType > other.PolarType)
        elif self.Angle != other.Angle:
            return (self.Angle > other.Angle)
        elif self.Thickness != other.Thickness:
            return (self.Thickness > other.Thickness)
        elif self.Strain != other.Strain:
            return (self.Strain > other.Strain)
        elif self.Grayscale != other.Grayscale:
            return (self.Grayscale > other.Grayscale)

    def __lt__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        if self.Color != other.Color:
            return (self.Color < other.Color)
        elif self.PolarType != other.PolarType:
            return (self.PolarType < other.PolarType)
        elif self.Angle != other.Angle:
            return (self.Angle < other.Angle)
        elif self.Thickness != other.Thickness:
            return (self.Thickness < other.Thickness)
        elif self.Strain != other.Strain:
            return (self.Strain < other.Strain)
        elif self.Grayscale != other.Grayscale:
            return (self.Grayscale < other.Grayscale)

    def __ne__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return not (self == other)

    def __ge__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return (self == other) or (self > other)

    def __le__(self, other):
        assert isinstance(other, SpectrumData), "ERROR: Data type not match!\n"
        return (self == other) or (self < other)

    @classmethod  #Direct class name. method name () to call, no self parameter is required, but the first parameter needs to be the cls parameter representing its own class
    def get_from_file(cls, FileName):  #cls is similar to self, you can use cls.xxxx to call the attributes and methods of the class. You can directly use the class name. Method name () to call
        """Get a new SpectrumData from file"""  #Get new SpectrumData from file
        assert os.path.isfile(FileName), "Fail to find file %s" % FileName

        # Get information from fullname    
        FullName = os.path.abspath(FileName)  #
        Info = {}
        NameSecs = FullName.split(os.sep)
        Info['Angle'] = float(45)
        Info['Color'] = NameSecs[4]
        Info['Thickness'] = float(NameSecs[2][0:2])
        Info['Strain'] = float(NameSecs[3][3:-1])
        Info['Grayscale'] = float(NameSecs[5][0])

        # Get information from text     
        lines = read_txt_lines(FileName)
        Idx_DataStart = None
        DataNames = {0: 'X'}  ## the name of XYDATA 
        for i, l in enumerate(lines):
            KeyList = l.split()
            if KeyList[0] == 'YUNITS':
                DataNames[1] = YName_Dict[KeyList[1]]
            elif KeyList[0] == 'Y2UNITS':
                DataNames[2] = YName_Dict[KeyList[1]]
            elif KeyList[0] == 'Y3UNITS':
                DataNames[3] = YName_Dict[KeyList[1]]
            elif KeyList[0] == 'XYDATA':
                Idx_DataStart = i + 1
                break
        # Initialize XYDATA
        for value in DataNames.values():
            Info[value] = []
        # read XYDATA
        for dataline in lines[Idx_DataStart:]:
            if dataline.startswith('[Comments]'):
                break
            for i, x in enumerate(dataline.split()):
                Info[DataNames[i]].append(float(x))
        # Arrange XYDATA in positive order
        idx_sorted = np.argsort(Info['X'])
        for key in DataNames.values():
            Info[key] = np.array(Info[key])[idx_sorted].tolist()
        return cls.get_from_dict(Info)

    def as_dict(self):
        return self.__dict__  #__dict__ can act on files, classes or class objects, and the final result is a dictionary.

    @classmethod
    def get_from_dict(cls, Dict):
        """Get a new SpectrumData from str line"""
        sd = cls()
        sd.__dict__.update(Dict)
        sd.Name = sd.default_name()
        return sd

    def default_name(self):
        """Default name of a SpectrumData"""
        return '%s-%s-%g-%g-%g-%g' % (self.Color, self.PolarType, self.Angle, self.Thickness, self.Strain, self.Grayscale)

    def __str__(self):
        # idstr = 'None' if self.ID is None else '%d' % self.ID
        idstr = str(self.ID)
        strtmp = 'ID: %s\nName: %s\nColor: %s\nPolarType: %s\n' % (idstr, self.Name, self.Color, self.PolarType)
        strtmp += 'Angle: %g°\nThickness: %g\nStrain: %g%%\nGrayscale: %g\n' % (self.Angle, self.Thickness, self.Strain, self.Grayscale)
        strtmp += 'X: %d values %g %g ... %g\n' % (len(self.X), self.X[0], self.X[1], self.X[-1])
        strtmp += 'G: %d values\nCD: %d values\nABS: %d values\n' % (len(self.G), len(self.CD), len(self.ABS))
        return strtmp


class SpectrumDataBase(object):
    """Database of Spectrum Data"""  #Spectral data information

    def __init__(self, PATH=''):
        self.Path = PATH
        self.DataName = []  # List of names of spectral data
        self.FullData = []  # All spectral data in the database

    def save(self):
        """Save database into files"""
        OutputData = {sd.Name: sd.as_dict() for sd in self.FullData}
        save_file(self.Path, OutputData)

    def load(self):
        """Load database from its path"""
        InputData = read_file(self.Path)
        Name, Data = [], []
        for k, v in InputData.items():
            Name.append(k)
            Data.append(SpectrumData.get_from_dict(v))
        self.DataName = Name
        self.FullData = Data
        return self

if __name__ == "__main__":
    #Call functions
    SourcePath = './'
    DataBasePath = './sdb.msgz'  # Your Database
    sdb = SpectrumDataBase(DataBasePath)
    Num = 0
    for TopDir, DirList, FileList in os.walk(SourcePath):
        for file in FileList:
            if file[-4:] != '.txt':
                continue
            sd = SpectrumData.get_from_file(os.path.join(TopDir, file))
            sd.ID = Num
            sd.PolarType = 'C'  # C means circular polarized light, L means linear polarized light
            sd.Name = sd.default_name()  # update data name
            sdb.DataName.append(sd.Name)
            sdb.FullData.append(sd)
            Num += 1
    sdb.save()
    # Test whether the data stored in the database is normal
    test = SpectrumDataBase(DataBasePath)
    test.load()
    for i in np.random.randint(0, len(test.DataName), 4):
        print('SpectrumData:', test.FullData[i])
