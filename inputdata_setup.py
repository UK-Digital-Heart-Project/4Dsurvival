import os, sys
import numpy as np
import pandas as pd
import fnmatch
import pickle


def all_files_exist(flist):
    numfiles = len(flist)
    allexist = True
    co = 0
    while allexist and co < numfiles:
        allexist = os.path.isfile(flist[co])
        co += 1
    return allexist

def file_len(fname):
    if os.path.isfile(fname) and os.path.getsize(fname)>0:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    else: 
        print('Failed to read {}!'.format(fname))
        return -1


try:
    stubdir = sys.argv[1]
    print('Reading mesh motion data from directory {}....'.format(stubdir))
except:
    print('Please pass name of directory containing segmented data...')


mpointsfile = 'matchedpointsnew.txt'
pim2 = 'subjnames.txt'
outcomefile = os.path.join(stubdir,'surv_outcomes.csv')
stepsuccess = [True for _ in range(3)]


#--Read mpointsfile
try: 
    mpoints = np.loadtxt(os.path.join(stubdir,mpointsfile), dtype=int)
except: 
    stepsuccess[0] = False
    print('{} read failed!'.format(mpointsfile))

#--Read list of subjects
try: 
    with open(os.path.join(stubdir,pim2)) as f: IDlist = [lin.strip('\n') for lin in f.readlines() if len(lin)>1]
except: 
    stepsuccess[1] = False
    print('{} read failed!'.format(pim2))

#--Find number of vertices
if stepsuccess[1]:
    try:
        meshtxtfile = os.path.join(stubdir,IDlist[0],'motion/RV_fr00.txt')
        num_vertx = file_len(meshtxtfile)
        if num_vertx <= 0:
            stepsuccess[2] = False
            print('There was a problem reading {} in order to determine number of vertices in 3D meshes!'.format(meshtxtfile))
    except:
        stepsuccess[2] = False
        print('Failed to read {} in order to determine number of vertices in 3D meshes!'.format(meshtxtfile))



validIDs = [False]
numframes = 20
if all(stepsuccess):
    print('\n\n------------------------------------------')
    print('Reading mesh motion data from directory {}...'.format(stubdir))
    print('Subject IDs will be read from file {}...'.format(pim2))
    print('Expected number of vertices per mesh = {0}, of which {1} will be extracted'.format(num_vertx, mpoints.shape[0]))
    print('Outcome data will be read from file {}...'.format(outcomefile))
    print('------------------------------------------\n\n\n')
    if os.path.exists(stubdir):
        if len(IDlist)>0:
            validIDs = [False for _ in range(len(IDlist))]
            X_all = np.zeros(shape=(len(IDlist),(numframes-1),mpoints.shape[0],3), dtype=float)
            for counter,ID in enumerate(IDlist):
                if os.path.exists(os.path.join(stubdir,ID)):
                    if os.path.exists(os.path.join(stubdir,ID,'motion')):
                        frames_file_list = [os.path.join(stubdir, ID, 'motion/RV_fr' + '{:0>2}'.format(b) + '.txt') for b in range(numframes)]
                        if all_files_exist(frames_file_list):
                            nframes = len(fnmatch.filter(os.listdir(os.path.join(stubdir , ID , 'motion')), 'RV_fr*.txt'))
                            if nframes==numframes:
                                if np.sum([file_len(frames_file_list[i])==num_vertx for i in range(numframes)])==numframes:
                                    vs = [True for _ in range(numframes)]
                                    try:
                                        coords_fr0 = np.loadtxt(frames_file_list[0])[mpoints[:,1]]
                                    except: 
                                        print('Error! could not read file {} !'.format(frames_file_list[0]))
                                        vs[0]=False
                                    if vs[0]:
                                        for j in range(1,numframes):
                                            try: 
                                                coords_frj = np.loadtxt(frames_file_list[j])[mpoints[:,1]]
                                            except: 
                                                print('Error! could not read file {} !'.format(frames_file_list[j]))
                                                vs[j]=False
                                            if vs[j]:
                                                X_all[counter,j-1,:,:] = coords_frj - coords_fr0
                                            else:
                                                break
                                        if np.all(vs):
                                            validIDs[counter] = True
                                            print('Successfully read motion data for ID {}'.format(ID))
                #               else: print(ID + ' RV files do not have ' + str(num_vertx) + ' vertices')
                                else: print('{0} : wrong # of vertices, expected {1} for all {2} frames but got {3}'.format(ID,num_vertx,numframes,str([file_len(frames_file_list[i]) for i in range(numframes)])))
                            else: print('{0} : RV files exist but not {1} in number. Skipping to next ID....'.format(ID,numframes))
                        else: print(ID + ' : folder exists but not all RV files exist. Skipping to next ID....')
                    else: print('There is no motion folder under directory {} !'.format(os.path.join(stubdir,ID)))
                else: print('{0} folder does not exist under directory {1}'.format(ID,stubdir))
        else: print('No IDs found in predinput_master2.txt !')
    else: print('directory meant to contain IDs is not valid!' )
else: pass


if any(validIDs): 
    numvalids = np.sum(validIDs)
    print('{} IDs with valid mesh motion data were found'.format(numvalids))
    X = X_all[validIDs]
else: print('No valid mesh motion data could be read!')


#Processing outcome data

#Read outcome master file - Column 1: ID, Column 2: censoring status, Column 3: time to event/censoring
#Tests of outcome file:
    #number of columns is 3
    #columns ordered correctly - ID, status, time
    #columns contain correct data (ID is string, status = 0 or 1, time > 0)
if any(validIDs):
    oreadable = True
    ofmtcorr1 = True
    if os.path.exists(outcomefile):
        try:
            outcome_df = pd.read_csv(outcomefile)
        except: 
            print('Error in reading outcome file {} !'.format(outcomefile))
            oreadable = False
        if oreadable:
            print('Outcome file {0} read: {1} rows and {2} columns...'.format(outcomefile, outcome_df.shape[0], outcome_df.shape[1]))
            if len(outcome_df.columns) != 3:
                print('Wrong number of columns in outcome file {} ! Expected 3 columns'.format(outcomefile))
            else:
                outcome_df.columns = ['ID','status','time']
                try: 
                    ocorrfmt = np.all([ i and j for (i,j) in zip([l in [0,1] for l in list(outcome_df.status)], [k>=0 for k in list(outcome_df.time)])])
                except: 
                    ofmtcorr1 = False
                    ocorrfmt = False
                if not (ofmtcorr1 and ocorrfmt):
                    print('status and/or time columns in {} are incorrectly formatted!'.format(outcomefile))
                    if ofmtcorr1==True and ocorrfmt==False:
                        aw = np.argwhere([ not(i and j) for (i,j) in zip([l in [0,1] for l in list(outcome_df.status)], [k>=0 for k in list(outcome_df.time)])])
                        if aw.shape[0] > 0:
                            print('{} {rw} {w} problematic: '.format(aw.shape[0],rw='rows' if aw.shape[0]>1 else 'row',w='were' if aw.shape[0]>1 else 'was'))
                            print(outcome_df.iloc[list(aw[:,0])])
                else: 
    #                pass
                    if any(validIDs):
                        print('matching mesh motion data IDs to outcome data IDs....')
                        IDlist_valids = list(np.array(IDlist)[validIDs])
                        #IDlist_woutc = [ii for ii in IDlist_valids if ii in list(outcome_df.ID)]
                        IDlist_woutc = list(set(list(outcome_df.ID)).intersection(set(IDlist_valids)))
                        if len(IDlist_woutc)==0:
                            print('None of the IDs from the mesh motion data were found in outcome file {}'.format(outcomefile))
                        else:
                            print('{1} of {2} valid IDs from mesh motion data were found in outcome file {0}'.format(outcomefile, len(IDlist_woutc), len(IDlist_valids)))
                            if len(IDlist_woutc) < len(IDlist_valids):
                                print('The following IDs from the mesh motion data were not found in outcome file {} :'.format(outcomefile))
                                print([ii for ii in IDlist_valids if ii not in list(outcome_df.ID)])
    #                       else:
                            y = outcome_df[(outcome_df['ID'].isin(IDlist_woutc))]
                            matchmask = [(u in IDlist_woutc) for u in IDlist_valids]
                            Xout = X[matchmask]
                            xshp = Xout.shape
                            xymatch = (y.shape[0]==xshp[0])
                            assert xymatch, 'ERROR: mesh motion (x) data has {1} rows while outcome (y) data has {0} rows'.format(y.shape[0], xshp[0])
                            if xymatch:
                                Xfin = Xout.reshape(xshp[:2]+(np.prod(xshp[2:]),)).reshape((xshp[0],-1))
                                plist = [Xfin,np.array(y[['status','time']]),list(y.ID)]
                                pklname = 'inputdata_DL' + '.pkl'
                                pklpath = os.path.join(os.getcwd(),'data',pklname)
                                with open(pklpath, 'wb') as f: pickle.dump(obj=plist, file=f)
                                print('Mesh motion and corresponding survival data for {0} subjects has been saved in {1}'.format(xshp[0],pklpath))
        else: pass
    else: print('Outcome file {} does not exist! Outcome data cannot be read!'.format(outcomefile))
