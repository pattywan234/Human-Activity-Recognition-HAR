"""
This file is for concatenate data.
"""
import numpy as np

# data
s2 = np.load('wesad/S2/Normalize/label_selected/label_4/all_extracted.npy')
s3 = np.load('wesad/S3/normalize/label_selected/label_4/all_extracted.npy')
s4 = np.load('wesad/S4/normalize/label_selected/label_4/all_extracted.npy')
s5 = np.load('wesad/S5/normalize/label_selected/label_4/all_extracted.npy')
s6 = np.load('wesad/S6/normalize/label_selected/label_4/all_extracted.npy')
s7 = np.load('wesad/S7/normalize/label_selected/label_4/all_extracted.npy')
s8 = np.load('wesad/S8/normalize/label_selected/label_4/all_extracted.npy')
s9 = np.load('wesad/S9/normalize/label_selected/label_4/all_extracted.npy')
s10 = np.load('wesad/S10/normalize/label_selected/label_4/all_extracted.npy')
s11 = np.load('wesad/S11/normalize/label_selected/label_4/all_extracted.npy')
s13 = np.load('wesad/S13/normalize/label_selected/label_4/all_extracted.npy')
s14 = np.load('wesad/S14/normalize/label_selected/label_4/all_extracted.npy')
s15 = np.load('wesad/S15/normalize/label_selected/label_4/all_extracted.npy')
s16 = np.load('wesad/S16/normalize/label_selected/label_4/all_extracted.npy')
s17 = np.load('wesad/S17/normalize/label_selected/label_4/all_extracted.npy')

# label
ls2 = np.load('wesad/S2/Normalize/label_selected/label_4.npy')[0:len(s2)]
ls3 = np.load('wesad/S3/raw/label_4.npy')[0:len(s3)]
ls4 = np.load('wesad/S4/raw/label_4.npy')[0:len(s4)]
ls5 = np.load('wesad/S5/raw/label_4.npy')[0:len(s5)]
ls6 = np.load('wesad/S6/raw/label_4.npy')[0:len(s6)]
ls7 = np.load('wesad/S7/raw/label_4.npy')[0:len(s7)]
ls8 = np.load('wesad/S8/raw/label_4.npy')[0:len(s8)]
ls9 = np.load('wesad/S9/raw/label_4.npy')[0:len(s9)]
ls10 = np.load('wesad/S10/raw/label_4.npy')[0:len(s10)]
ls11 = np.load('wesad/S11/raw/label_4.npy')[0:len(s11)]
ls13 = np.load('wesad/S13/raw/label_4.npy')[0:len(s13)]
ls14 = np.load('wesad/S14/raw/label_4.npy')[0:len(s14)]
ls15 = np.load('wesad/S15/raw/label_4.npy')[0:len(s15)]
ls16 = np.load('wesad/S16/raw/label_4.npy')[0:len(s16)]
ls17 = np.load('wesad/S17/raw/label_4.npy')[0:len(s17)]

# uls2, cls2 = np.unique(ls2, return_counts=True)
# print(dict(zip(uls2, cls2)))
# uls3, cls3 = np.unique(ls3, return_counts=True)
# print(dict(zip(uls3, cls3)))
# uls4, cls4 = np.unique(ls4, return_counts=True)
# print(dict(zip(uls4, cls4)))
# uls5, cls5 = np.unique(ls5, return_counts=True)
# print(dict(zip(uls5, cls5)))
# uls6, cls6 = np.unique(ls6, return_counts=True)
# print(dict(zip(uls6, cls6)))
# uls7, cls7 = np.unique(ls7, return_counts=True)
# print(dict(zip(uls7, cls7)))
# uls8, cls8 = np.unique(ls8, return_counts=True)
# print(dict(zip(uls8, cls8)))
# uls9, cls9 = np.unique(ls9, return_counts=True)
# print(dict(zip(uls9, cls9)))
# uls10, cls10 = np.unique(ls10, return_counts=True)
# print(dict(zip(uls10, cls10)))
# uls11, cls11 = np.unique(ls11, return_counts=True)
# print(dict(zip(uls11, cls11)))
# uls13, cls13 = np.unique(ls13, return_counts=True)
# print(dict(zip(uls13, cls13)))
# uls14, cls14 = np.unique(ls14, return_counts=True)
# print(dict(zip(uls14, cls14)))
# uls15, cls15 = np.unique(ls15, return_counts=True)
# print(dict(zip(uls15, cls15)))
# uls16, cls16 = np.unique(ls16, return_counts=True)
# print(dict(zip(uls16, cls16)))
# uls17, cls17 = np.unique(ls17, return_counts=True)
# print(dict(zip(uls17, cls17)))

# all data
data_after_concat = np.concatenate((s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s13, s14, s15, s16, s17))
label_after_concat = np.concatenate((ls2, ls3, ls4, ls5, ls6, ls7, ls8, ls9, ls10, ls11, ls13, ls14, ls15, ls16, ls17))

# 10 subjects for training and 5 subjects for test the model


np.save('wesad/all_subjects/label_4/data_all_subject.npy', data_after_concat)
np.save('wesad/all_subjects/label_4/label_all_subject.npy', label_after_concat)

print('DONE!!')
