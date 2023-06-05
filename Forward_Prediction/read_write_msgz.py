import gzip
import spectrum as sp
# dye_property = sp.read_file('./Data/dye_ABS.msgz')   #different colored abs
sdb_statistics = sp.read_file('./Data/sdb_XG.msgz')
print(sdb_statistics)

for key in sdb_statistics:
    sdb_statistics[key].pop('CD', None)
    sdb_statistics[key].pop('ABS', None)
print(sdb_statistics)   
sp.save_file('./Data/sdb_XG.msgz',sdb_statistics)




