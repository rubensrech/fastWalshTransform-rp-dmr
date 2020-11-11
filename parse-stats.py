import re
import sys
import subprocess

dir = sys.argv[1]

def getStats(filename):
    file = open(filename)
    stats = {}
    for line in file:
        sciNotRegex = '([-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?)'
        
        maxAbsErrRegex = re.compile('> Max absolute err: ' + sciNotRegex)
        minAbsErrRegex = re.compile('> Min absolute err: ' + sciNotRegex)
        avgAbsErrRegex = re.compile('> Avg absolute err: ' + sciNotRegex)

        maxAbsErrMatches = maxAbsErrRegex.search(line)
        minAbsErrMatches = minAbsErrRegex.search(line)
        avgAbsErrMatches = avgAbsErrRegex.search(line)
        
        if maxAbsErrMatches: 
            stats['maxAbsErr'] = maxAbsErrMatches.group(1)

        if minAbsErrMatches: 
            stats['minAbsErr'] = minAbsErrMatches.group(1)

        if avgAbsErrMatches: 
            stats['avgAbsErr'] = avgAbsErrMatches.group(1)

    return stats

def findInjPathsByClassification(classification):
    resp = subprocess.check_output('grep -rl "' + classification + '" ' + dir + '/*/stdout.txt', shell=True)
    paths = resp.split('\n')
    paths.pop() # remove empty item
    for (i, p) in enumerate(paths):
        t = p.split('/')
        t.pop()
        t = '/'.join(t)
        paths[i] = t
    return paths

def findInjPathsForTP():
    return findInjPathsByClassification('TRUE POSITIVE')

def findInjPathsForFN():
    return findInjPathsByClassification('FALSE NEGATIVE')

pathsTP = findInjPathsForTP()
pathsFN = findInjPathsForFN()

print("=================================================================")
print("== TRUE POSITIVES ===============================================")
for path in pathsTP:
    stats = getStats(path + '/out-vs-gold-stats.txt')
    print("    " + str(stats))

print("=================================================================")
print("== FALSE NEGATIVES ==============================================")
for path in pathsFN:
    stats = getStats(path + '/out-vs-gold-stats.txt')
    print("    " + str(stats))
