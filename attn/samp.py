RANDOM = 'random'
COMMON = 'common'
OVERLAP = 'overlap'
CONCATE = 'concate'

def wzt_sample1(step=0):
    return {RANDOM:0.5,COMMON:0.5}
def wzt_sample2(step=0):
    return {RANDOM:0.2,COMMON:0.8}
def wzt_sample3(step=0):
    return {RANDOM:0.3,COMMON:0.5,OVERLAP:0.2}
def wzt_sample4(step=0):
    return {RANDOM:0.3,COMMON:0.5,CONCATE:0.2}
def wzt_sample5(step=0):
    return {RANDOM:0.25,COMMON:0.25,OVERLAP:0.25,CONCATE:0.25}