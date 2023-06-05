#!/usr/bin/env python
'''
Author: Ifsoul
Date: 2020-08-31 10:32:25
LastEditTime: 2022-04-09 12:18:31
LastEditors: Ifsoul
Description: read uv-vis data
'''

import os
import sys
import time
import numpy as np
import ujson
import zlib
import msgpack
from chardet import detect

ColorList = (
    'k',
    'xkcd:red',
    'xkcd:bright blue',
    'xkcd:green',
    'xkcd:purple',
    'xkcd:pink',
    'xkcd:brown',
    'xkcd:orange',
    'xkcd:aqua',
    'xkcd:olive',
)
ShapeList = ('o', 's', '^', 'v', '>', '<', 'D', '*', 'P', 'X')

if msgpack.version[0] >= 1:
    msg_load_kwarg = {'strict_map_key': False}
else:
    msg_load_kwarg = {'encoding': 'utf-8'}


def save_as_json(filename, something):
    """Save something as json into file""" 
    with open(filename, 'wt') as f:
        ujson.dump(something, f)  


def load_from_json(filename):
    """Load something from json file""" 
    if not os.path.exists(filename):  
        raise AssertionError('File not found: %s' % filename)
    with open(filename, 'rt') as f:
        something = ujson.load(f) 
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
    """Read lines from txt file, skipping blank and comment lines. Return a list.""" 
    with open(FileName, 'rb') as f:
        cur_encoding = detect(f.read(10000))['encoding']
    with open(FileName, 'rt', encoding=cur_encoding) as f:
        lines = f.readlines()
    TreatMethod = str.split if SPLIT else str.strip
    return list(TreatMethod(l) for l in lines if l.strip() and (IncludeComment or l.strip()[0] != '#'))


def get_num_from_str(str_in):
    last_num = 0
    for i, s in enumerate(str_in):
        if s.isdigit() or s in '.-':
            last_num = i
        else:
            break
    # print(str_in)
    return float(str_in[:last_num + 1])


def int2roman(num: int) -> str:
    a = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    b = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    res = ''
    for i, n in enumerate(a):
        while num >= a[i]:
            res += b[i]
            num -= a[i]
    return res


def roman2int(s: str) -> int:
    numbers = {
        'I': 1,
        'IV': 5,
        'V': 5,
        'IX': 10,
        'X': 10,
        'XL': 50,
        'L': 50,
        'XC': 100,
        'C': 100,
        'CD': 500,
        'D': 500,
        'CM': 1000,
        'M': 1000,
    }
    sum = 0
    n = 0
    while n <= len(s) - 1:
        if (numbers.get(s[n:n + 2])) != None and len(s[n:n + 2]) >= 2:
            sum = sum + numbers.get(s[n:n + 2]) - numbers[s[n]]
            n = n + 2
        else:
            sum = sum + numbers[s[n]]
            n = n + 1
    return sum


def str_sep(abc123):
    '''Cut a string like "abc123" into ["abc","123"]'''
    for c in abc123[::-1]:
        if not c.isdigit():
            break
    pos = abc123.rfind(c) + 1
    return abc123[:pos], abc123[pos:]


def sec2time(sec):
    '''Transform seconds into readable time format'''
    if isinstance(sec, str):
        sec = float(sec)
    assert isinstance(sec, int) or isinstance(sec, float), "ERROR: Input should be a number.\n"
    t = ' %gs' % (sec % 60)
    if sec >= 60:
        sec = sec // 60
        t = ' %dm' % (sec % 60) + t
        if sec >= 60:
            sec = sec // 60
            t = ' %dh' % (sec % 60) + t
            if sec >= 24:
                sec = sec // 24
                t = ' %dd' % (sec) + t
    return t


class timer(object):

    def __init__(self, name='', outfile=sys.stdout):
        self.name = name
        self.output = outfile

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print('%s time: %s' % (self.name, sec2time(self.end - self.start)), file=self.output)


def get_para(Arg_List, Num_Dict={'.': -1}, user_help_key=False, help=lambda: None):
    '''Get parameters from argument list'''
    if not user_help_key:
        Num_Dict.update({'h': 0, 'help': 0})
    para_pos = [i for i, p in enumerate(Arg_List) if p.startswith('-') and p[1:] in Num_Dict]
    para_Dict = {}
    for key in Num_Dict:
        if isinstance(Num_Dict[key], int):
            Num_Dict[key] = (Num_Dict[key], Num_Dict[key])
        elif Num_Dict[key][1] < 0:
            Num_Dict[key] = (Num_Dict[key][0], len(Arg_List))
    if '.' in Num_Dict:
        Num1 = para_pos[0] if para_pos else len(Arg_List)
        if Num_Dict['.'][0] >= 0 and not Num_Dict['.'][0] <= Num1 <= Num_Dict['.'][1]:
            print("ERROR: Unmatched number of parameters!")
            print("Expect %d%s parameter(s) but %d got.\n" % (Num_Dict['.'][0], '' if Num_Dict['.'][1] == Num_Dict['.'][0] else ' to %d' % Num_Dict['.'][1], Num1))
            help()
            sys.exit()
        else:
            para_Dict['.'] = Arg_List[:Num1]
    for k, i in enumerate(para_pos):
        key = Arg_List[i][1:]
        pos_st = i + 1
        pos_ed = para_pos[k + 1] if k + 1 < len(para_pos) else len(Arg_List)
        if Num_Dict[key][0] >= 0 and not Num_Dict[key][0] <= pos_ed - pos_st <= Num_Dict[key][1]:
            print("ERROR: Unmatched number of parameters for option -%s!" % key)
            print("Expect %d%s parameter(s) but %d got.\n" % (Num_Dict[key][0], '' if Num_Dict[key][1] == Num_Dict[key][0] else ' to %d' % Num_Dict[key][1], pos_ed - pos_st))
            help()
            sys.exit()
        para_Dict[key] = Arg_List[pos_st:pos_ed]
    if not user_help_key:
        if 'h' in para_Dict or 'help' in para_Dict:
            help()
            sys.exit()
    return para_Dict


class UVSpectrum(object):
    """Informations of Spectrum Data""" 

    def __init__(self, ID=None, NAME='', FILM='', COLOR='', THICK=0, STRAIN=0, DYETIME=0, GRAY=-1):
        self.ID = ID
        self.Name = NAME
        self.Film = FILM
        self.Color = COLOR  
        self.Thickness = THICK  
        self.Strain = STRAIN 
        self.DyeTime = DYETIME 
        self.GrayScale = GRAY  
        self.X = []
        self.T0 = []
        self.T90 = []
        self.Ld = []

    def __eq__(self, other):
        assert isinstance(other, UVSpectrum), "ERROR: Data type not match!\n"
        return (self.Name == other.Name)

    def __gt__(self, other):
        assert isinstance(other, UVSpectrum), "ERROR: Data type not match!\n"
        if self.Film != other.Film:
            return (self.Film > other.Film)
        elif self.Color != other.Color:
            return (self.Color > other.Color)
        elif self.Thickness != other.Thickness:
            return (self.Thickness > other.Thickness)
        elif self.Strain != other.Strain:
            return (self.Strain > other.Strain)
        elif self.DyeTime != other.DyeTime:
            return (self.DyeTime > other.DyeTime)

    def __lt__(self, other):
        assert isinstance(other, UVSpectrum), "ERROR: Data type not match!\n"
        if self.Film != other.Film:
            return (self.Film < other.Film)
        elif self.Color != other.Color:
            return (self.Color < other.Color)
        elif self.Thickness != other.Thickness:
            return (self.Thickness < other.Thickness)
        elif self.Strain != other.Strain:
            return (self.Strain < other.Strain)
        elif self.DyeTime != other.DyeTime:
            return (self.DyeTime < other.DyeTime)

    def __ne__(self, other):
        assert isinstance(other, UVSpectrum), "ERROR: Data type not match!\n"
        return not (self == other)

    def __ge__(self, other):
        assert isinstance(other, UVSpectrum), "ERROR: Data type not match!\n"
        return (self == other) or (self > other)

    def __le__(self, other):
        assert isinstance(other, UVSpectrum), "ERROR: Data type not match!\n"
        return (self == other) or (self < other)

    @staticmethod
    def read_info_from_filename(FullName, MemType=0):
        """Read experimental parameters from filename"""
        Info = {}
        # print(FullName)
        NameSecs = FullName[:FullName.rfind('.')].split(os.sep)
        if MemType == 0:
            for l in NameSecs:
                if 'um' in l and l[0].isdigit():
                    Info['Thickness'] = float(l[:l.find('um')])
                if '%' in l or '％' in l:
                    key_strs = [_ for _ in l.split('-') if _]
                    name = key_strs[-4]
                    if name not in ('blue2', 'blue6', 'blue71'):
                        name, gray = str_sep(name)
                    else:
                        gray = -1
                    Info['Color'] = name
                    Info['GrayScale'] = gray
                    Info['PARALLEL'] = True if key_strs[-1] == '0' else False
                    Info['Strain'] = get_num_from_str(key_strs[-2])
                    Info['DyeTime'] = get_num_from_str(key_strs[-3])
        else:
            for l in NameSecs:
                if 'um' in l:
                    key_strs = l.split('-')
                    Info['Film'] = key_strs[0]
                    thick_str = key_strs[-1]
                    Info['Thickness'] = float(thick_str[:thick_str.find('um')])
            key_strs = [_ for _ in NameSecs[-1].split('-') if _]
            Info['PARALLEL'] = True if (key_strs[-1] == '0' or key_strs[-1] == '45') else False
            Info['Strain'] = get_num_from_str(key_strs[-2])
        return Info

    @classmethod  
    def get_from_file(cls, FileName, FullName='', MemType=0): 
        """Get a new UVSpectrum from file"""  
        assert os.path.isfile(FileName), "Fail to find file %s" % FileName

        # Get information from fullname   
        if not FullName:
            FullName = os.path.abspath(FileName)  #
        Info = cls.read_info_from_filename(FullName, MemType=MemType)

        print(FullName)
        print(Info)
        # Get information from text    
        lines = read_txt_lines(FileName)
        DataNames = {0: 'X'} 
        DataNames[1] = 'T0' if Info['PARALLEL'] else 'T90'
        # 初始化 XYDATA
        for value in DataNames.values():
            Info[value] = []
        # 读取 XYDATA
        for dataline in lines[2:]:
            if dataline.startswith('[Comments]'):
                break
            for i, x in enumerate(dataline.split()):
                Info[DataNames[i]].append(float(x))
        # 正序排列 XYDATA
        idx_sorted = np.argsort(Info['X'])
        for key in DataNames.values():
            Info[key] = np.array(Info[key])[idx_sorted].tolist()
        Info.pop('PARALLEL')

       
        return cls.get_from_dict(Info)

    def as_dict(self):
        return self.__dict__ 

    @classmethod
    def get_from_dict(cls, Dict):
        """Get a new UVSpectrum from str line"""
        sd = cls()
        sd.__dict__.update(Dict)
        sd.Name = sd.default_name()
        return sd

    def default_name(self):
        """Default name of a UVSpectrum"""
        return '%s:%s:%g:%g:%g' % (self.Film, self.Color, self.Thickness, self.Strain, self.DyeTime)

    def __str__(self):
        # idstr = 'None' if self.ID is None else '%d' % self.ID
        idstr = str(self.ID)
        strtmp = 'ID: %s\nName: %s\nFilm: %s\nColor: %s\n' % (idstr, self.Name, self.Film, self.Color)
        strtmp += 'Thickness: %g\nStrain: %g%%\nDyeTime: %g\n' % (self.Thickness, self.Strain, self.DyeTime)
        strtmp += 'X: %d values %g %g ... %g\n' % (len(self.X), self.X[0], self.X[1], self.X[-1])
        strtmp += 'T0: %d values\nT90: %d values\n' % (len(self.T0), len(self.T90))
        return strtmp


class UVSpectrumDataBase(object):
    """Database of Spectrum Data"""  

    def __init__(self, PATH=''):
        self.Path = PATH
        self.DataName = []  
        self.FullData = []  

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
            Data.append(UVSpectrum.get_from_dict(v))
        self.DataName = Name
        self.FullData = Data
        return self
