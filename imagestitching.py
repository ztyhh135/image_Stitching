from __future__ import print_function  #
import cv2
import argparse
import os
import numpy as np
from numpy import linalg


def knnmatch(des1,des2):
    matches=[]
    for i in range(des1.shape[0]):
        mindist = 1
        mindist_2 = 1
        minj = 0
        minj_2=0
        for j in range(des2.shape[0]):
            dist = np.sqrt(np.sum(np.square(des1[i] - des2[j])))
            if dist < mindist:
                mindist_2=mindist
                mindist = dist
                minj_2=minj
                minj = j
        if mindist < 0.3:
            matches.append([cv2.DMatch(i,minj,0,mindist),cv2.DMatch(i,minj_2,0,mindist_2)])
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return good

def findhomography(src, dst, itr,threshold=3):
    t = np.r_[0:src.shape[0]]
    goodcountmax = 0
    Hmax = np.zeros([3,3],dtype="float32")
    for i in range(itr):
        sample=np.random.choice(t,4,replace = False)
        src_sample = src[sample]
        dst_sample = dst[sample]
        A = np.array([[-src_sample[0][0],-src_sample[0][1],-1,0,0,0,src_sample[0][0]*dst_sample[0][0],src_sample[0][1]*dst_sample[0][0],dst_sample[0][0]],
                      [0,0,0,-src_sample[0][0],-src_sample[0][1],-1,src_sample[0][0]*dst_sample[0][1],src_sample[0][1]*dst_sample[0][1],dst_sample[0][1]],
                      [-src_sample[1][0],-src_sample[1][1],-1,0,0,0,src_sample[1][0]*dst_sample[1][0],src_sample[1][1]*dst_sample[1][0],dst_sample[1][0]],
                      [0,0,0,-src_sample[1][0],-src_sample[1][1],-1,src_sample[1][0]*dst_sample[1][1],src_sample[1][1]*dst_sample[1][1],dst_sample[1][1]],
                      [-src_sample[2][0],-src_sample[2][1],-1,0,0,0,src_sample[2][0]*dst_sample[2][0],src_sample[2][1]*dst_sample[2][0],dst_sample[2][0]],
                      [0,0,0,-src_sample[2][0],-src_sample[2][1],-1,src_sample[2][0]*dst_sample[2][1],src_sample[2][1]*dst_sample[2][1],dst_sample[2][1]],
                      [-src_sample[3][0],-src_sample[3][1],-1,0,0,0,src_sample[3][0]*dst_sample[3][0],src_sample[3][1]*dst_sample[3][0],dst_sample[3][0]],
                      [0,0,0,-src_sample[3][0],-src_sample[3][1],-1,src_sample[3][0]*dst_sample[3][1],src_sample[3][1]*dst_sample[3][1],dst_sample[3][1]]])
        U,sigma,VT=linalg.svd(A)
        H=VT.T[:,8].reshape(3,3)
        goodcount = 0
        for j in range(src.shape[0]):
            err = np.square((H[0][0]*src[j][0]+H[0][1]*src[j][1]+H[0][2])/(H[2][0]*src[j][0]+H[2][1]*src[j][1]+H[2][2])-dst[j][0])+ np.square((H[1][0]*src[j][0]+H[1][1]*src[j][1]+H[1][2])/(H[2][0]*src[j][0]+H[2][1]*src[j][1]+H[2][2])-dst[j][1])
            if err < threshold:
                goodcount+=1
        if goodcount>goodcountmax:
            goodcountmax = goodcount
            Hmax = H
        if goodcount>= 0.99*src.shape[0]:
            break
    return Hmax

def warpperspective(src,H):
    dst = np.zeros([int(src.shape[0]*2),int(src.shape[1]*2),3],dtype = "uint8")
    for i in range(dst.shape[0]):
#            if i%100==0:
#                print(i)
        for j in range(dst.shape[1]):
            dx = int((H[0][0]*i+H[0][1]*j+H[0][2])/(H[2][0]*i+H[2][1]*j+H[2][2]))
            dy = int((H[1][0]*i+H[1][1]*j+H[1][2])/(H[2][0]*i+H[2][1]*j+H[2][2]))
            if dx < src.shape[0] and dy < src.shape[1] and dx>=0 and dy>=0:
                dst[i][j] = src[dx][dy]
    return dst

def up_to_step_1(imgs):
    """Complete pipeline up to step 3: Detecting features and descriptors"""
    # ... your code here ...
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000,
                                           nOctaves = 4,
                                           nOctaveLayers = 3,
                                           upright = False,
                                           extended = False)
    imgs_out=[]
    for each in imgs:
        gray= cv2.cvtColor(each,cv2.COLOR_BGR2GRAY)
        kp = detector.detect(gray,None)
        imgs_out.append(cv2.drawKeypoints(each, kp, None,-1,4))
    imgs=imgs_out
    return imgs


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    # ... your code here ...
    i=0
    for each in imgs:
        i+=1
        cv2.imwrite(output_path+"/output"+str(i)+".jpg", each)


def up_to_step_2(imgs):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    # ... your code here ...
    matchlist=[]
    imgs_out=[]
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000,
                                           nOctaves = 4,
                                           nOctaveLayers = 3,
                                           upright = False,
                                           extended = False)
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i!=j:
                gray1= cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
                kp1,des1 = detector.detectAndCompute(gray1,None)
                gray2= cv2.cvtColor(imgs[j],cv2.COLOR_BGR2GRAY)
                kp2,des2 = detector.detectAndCompute(gray2,None)
                matches = knnmatch(des1,des2)
                matchlist.append([i,len(kp1),j,len(kp2),len(matches)])
                imgs_out.append(cv2.drawMatches(imgs[i], kp1,imgs[j], kp2,matches, None))
    imgs = imgs_out
    return imgs, matchlist


def save_step_2(imgs, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    # ... your code here ...
    for i in range(len(imgs)):
        name1,tail1 = str.split(filenames[match_list[i][0]],".")
        name2,tail2 = str.split(filenames[match_list[i][2]],".")
        cv2.imwrite(output_path+"/"+name1+"_"+str(match_list[i][1])+"_"+name2+"_"+str(match_list[i][3])+"_"+str(match_list[i][4])+".jpg", imgs[i])


def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    # ... your code here ...
    img_pairs=[]
    matchlist=[]
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i!=j:
                detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000,
                                                       nOctaves = 4,
                                                       nOctaveLayers = 3,
                                                       upright = False,
                                                       extended = False)
                gray1= cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
                kp1,des1 = detector.detectAndCompute(gray1,None)
                gray2= cv2.cvtColor(imgs[j],cv2.COLOR_BGR2GRAY)
                kp2,des2 = detector.detectAndCompute(gray2,None)
                matches = knnmatch(des1,des2)
    #            bf = cv2.BFMatcher()
    #            matches = bf.knnMatch(des1,des2, k=2)
    #            good = []
    #            for m,n in matches:
    #                if m.distance < 0.75*n.distance:
    #                    good.append(m)
                if len(matches)<10:
                    break
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
                H = findhomography(src_pts, dst_pts, 3000)
            #    H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)
                warp = warpperspective(imgs[i],H)
#                warp = cv2.warpPerspective(imgs[i], H, (imgs[i].shape[1]*2 , imgs[i].shape[0]*2))
            #    imgs = warp
            #    warp[0:imgs[0].shape[0], 0:imgs[0].shape[1]] = imgs[2]
                rows, cols = np.where(warp[:,:,0] !=0)
                min_row, max_row = min(rows), max(rows) +1
                min_col, max_col = min(cols), max(cols) +1
                warp_1 = warp[min_row:max_row,min_col:max_col,:]
                
                matches_2 = knnmatch(des2,des1)
                src_pts_2 = np.float32([ kp2[m.queryIdx].pt for m in matches_2 ])
                dst_pts_2 = np.float32([ kp1[m.trainIdx].pt for m in matches_2 ])
                H_2 = findhomography(src_pts_2, dst_pts_2, 3000)
                warp_2 = warpperspective(imgs[j],H_2)
#                warp_2 = cv2.warpPerspective(imgs[j], H_2, (imgs[j].shape[1]*2 , imgs[j].shape[0]*2))
                rows, cols = np.where(warp_2[:,:,0] !=0)
                min_row, max_row = min(rows), max(rows) +1
                min_col, max_col = min(cols), max(cols) +1
                warp_2 = warp_2[min_row:max_row,min_col:max_col,:]
                img_pairs.append([warp_1,warp_2])
                matchlist.append([i,j])
    imgs=img_pairs
    return imgs, matchlist


def save_step_3(img_pairs, match_list, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    # ... your code here ...
    for i in range(len(img_pairs)):
        name1,tail1 = str.split(filenames[match_list[i][0]],".")
        name2,tail2 = str.split(filenames[match_list[i][1]],".")
        cv2.imwrite(output_path+"/"+name1+"_"+name2+".jpg", img_pairs[i][0])
        cv2.imwrite(output_path+"/"+name2+"_"+name1+".jpg", img_pairs[i][1])


def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...
    for i in range(len(imgs)-1):
        
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000,
                                               nOctaves = 4,
                                               nOctaveLayers = 3,
                                               upright = False,
                                               extended = False)
        gray1= cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
        kp1,des1 = detector.detectAndCompute(gray1,None)
        gray2= cv2.cvtColor(imgs[i+1],cv2.COLOR_BGR2GRAY)
        kp2,des2 = detector.detectAndCompute(gray2,None)
#        bf = cv2.BFMatcher()
        matches = knnmatch(des2,des1)
#        good = []
#        for m,n in matches:
#            if m.distance < 0.75*n.distance:
#                good.append(m)
#                
        src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ])
        H = findhomography(src_pts, dst_pts, 3000)
#        H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)
    #    warp = warpperspective(imgs[0],H)
        warp = cv2.warpPerspective(imgs[i+1], H, (imgs[i+1].shape[1]*2 , imgs[i+1].shape[0]*2))
        rows, cols = np.where(warp[:,:,0] !=0)
        min_row, max_row = min(rows), max(rows) +1
        min_col, max_col = min(cols), max(cols) +1
        result = warp[min_row:max_row,min_col:max_col,:]
    #    imgs = warp
    #    warp[0:imgs[0].shape[0], 0:imgs[0].shape[1]] = imgs[2]
        stitcher = cv2.createStitcher(False)
        result = stitcher.stitch((imgs[i],result))
        imgs[i+1] = result[1]
    imgs[0] = imgs[-2]
    return imgs[0]


def save_step_4(imgs, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...
    cv2.imwrite(output_path+"/output.jpg", imgs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    filenames=[]
    for filename in os.listdir(args.input):
        print(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        filenames.append(filename)
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(imgs)
        save_step_2(modified_imgs, match_list, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs, match_list = up_to_step_3(imgs)
        save_step_3(img_pairs, match_list, args.output)
    elif args.step == 4:
        print("Running step 4")
        panoramic_img = up_to_step_4(imgs)
        save_step_4(panoramic_img, args.output)
