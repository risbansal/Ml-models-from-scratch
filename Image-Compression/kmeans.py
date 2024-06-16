from PIL import Image
import numpy as np
import time
import argparse
#time

def Kmeans(means, pixm, flag):
    ncents = []
    ml = []
    for m in means:
        pixnorm = pixm - np.array(m)
        pixnorm = np.apply_along_axis(np.linalg.norm, 2, pixnorm)
        #pixnorm = np.linalg.norm(pixnorm, axis = 2)
        ml.append(pixnorm)
    ml = np.array(ml)
    result = np.argmin(ml, axis=0)
    pixm_r = pixm.reshape(pixm.shape[0]*pixm.shape[1], pixm.shape[2])
    for i in range(int(ml.shape[0])):
        c = (result == i)
        d = c.reshape(pixm.shape[0]*pixm.shape[1], 1)
        e = np.repeat(d, 3, axis = 1)
        e = np.invert(e)
        f = np.ma.array(pixm_r, mask= e)
        ncents.append(np.mean(f, axis = 0).data)
        if flag == True:
            e = np.invert(e)
            pixm_r[:, 0][e[:, 0]] = int(means[i][0])
            pixm_r[:, 1][e[:, 1]] = int(means[i][1])
            pixm_r[:, 2][e[:, 2]] = int(means[i][2])

    if flag == True:
        pixm_c = pixm_r.reshape([pixm.shape[0], pixm.shape[1], pixm.shape[2]])
        return pixm_c
    else:
        return [list(x) for x in ncents]

def main():

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', '--img', type=str)
    parser.add_argument('-k', '--k', type=str)

    arg = parser.parse_args()
    img_name = arg.img
    img = img_name + '.jpg'
    K = int(arg.k)

    image = Image.open(img)

    width, height = image.size
    image.load()

    pixel_mat = np.asarray(image, dtype= "int32")

    flag = False
    converged = False
  
    prev_means = []

    for i in range(K):
        prev_means.append([np.random.randint(256), np.random.randint(256), np.random.randint(256)])

    print("Initial random Means : ", prev_means)

    for i in range(500):
        if (i+1) % 50 == 0:
            print("{}th epoch".format(i+1))
        new_means = Kmeans(prev_means, pixel_mat, flag)
        if prev_means == new_means:
            converged = True
            break
        else:
            prev_means = new_means
  
    flag = True
    new_mat = Kmeans(prev_means, pixel_mat, flag)
    new_mat = new_mat.astype(np.uint8)
    image_new = Image.fromarray(new_mat, "RGB")
    image_new.save("{}_K={}_E={}.png".format(img_name, K, i + 1))
    if converged:
        print("Kmeans converged at {} epoch ".format(i + 1))
    else:
        print("Kmeans not converged, algo ran for {} epochs".format(i + 1))
    etime = time.time() - start_time
    print("Time taken : {}".format(etime))

main()
