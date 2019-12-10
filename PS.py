import numpy,scipy.sparse
from sparsesvd import sparsesvd
import scipy
from scipy.linalg import svd
from numpy.linalg import matrix_rank
import numpy as np
from numpy import linalg as LA
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
import scipy.misc
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nameroot = "Image_"
images = []
for i in range(1,21):
    if i<10:
        stri = "0"+str(i)
    else:
        stri = str(i)
    imname = r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\PhotometricStereo\Images_Cat\Image_"+stri+".png"
    img = imread(imname)
    imsize = img.shape
    images.append(img)

#print(len(images))
imlength = images[0].shape[0]*images[0].shape[1]
nimages = len(images)
I = np.empty((0,imlength))

for img in images:
    i = img.flatten()
    I = np.append(I,[i],axis=0)

(U,s,vh) = np.linalg.svd(I, full_matrices=False)

L = U[:,0:3]
N = vh[0:3,:]
S_sqrt = np.diag(np.sqrt(s[:3]))
L = np.dot(L, S_sqrt)
N = np.dot(S_sqrt, N)

L_help = None
#print (L)
for i in range(0, L.shape[0]):
    x = L[i,0]
    y = L[i,1]
    z = L[i,2]
    arr = [x*x, 2*x*y, 2*x*z, y*y, 2*y*z, z*z]

    if L_help is None:
        L_help = np.array(arr)

    else:
        L_help = np.vstack((L_help,arr))

(b_p,res,rank,s) = np.linalg.lstsq(L_help,np.ones(L.shape[0]),rcond=None)

#print (b_p)

B = np.array([[b_p[0], b_p[1], b_p[2]],
              [b_p[1], b_p[3], b_p[4]],
              [b_p[2], b_p[4], b_p[5]]])

(U,s,vh) = np.linalg.svd(B)
A = np.dot(U,np.diag(np.sqrt(s)))

L = np.dot(L,A)
N = np.linalg.solve(A,N)
Rot = np.array([[1,  0, 0],
		[0, 1, 0],
		[0, 0, 1]])

L = np.dot(L,Rot)
N = np.linalg.solve(Rot,N)
#print(L.shape)
#print(N)

threshold = 100
i=0

normalsx = np.zeros((images[0].shape))
normalsy = np.zeros((images[0].shape))
normalsz = np.zeros((images[0].shape))
normal = np.zeros((images[0].shape))
normalrgb = np.zeros((images[0].shape[0], images[0].shape[1],3),dtype='uint8')

albedo = np.zeros((images[0].shape[0],images[0].shape[1],3))
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
'''
#print(normals)
imx = images[0].shape[0]
imy = images[0].shape[1]
ivec = np.zeros((len(images), 3))

for x in range(0,imx):
    for y in range(0,imy):
        norm = N[:,i]
        #print(np.cross(norm))
        #norm = norm/(np.linalg.norm(norm))
        #print(norm)
        if not np.isnan(np.sum(norm)):
            #print(norm)
            it = np.dot(L,norm)
            k = np.dot(np.transpose(ivec), it)/(np.dot(it, it))
            #print(k)
            if not np.isnan(np.sum(k)):
                albedo[x][y] = k
            normalsx[x][y] = abs(norm[0]*150)
            normalsy[x][y] = abs(norm[1]*150)
            normalsz[x][y] = abs(norm[2]*150)
            normal[x][y] = math.sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2])*150
            normalrgb[x][y] = [abs(norm[0]*160), abs(norm[1]*160), abs(norm[2]*160)]
            '''
            if x%5 ==0 and y%5 == 0:
                q = ax.quiver(y,1,x,norm[0],norm[1],norm[2],length=.01) #plot a quiver for v(x,y) at point x,y (V is inverted because the image vertical axis is inverted)
            '''
        i = i+1
#plt.show()
'''    
for (xT, value) in np.ndenumerate(images):
    if(value > threshold):
        normal = N[:, i]
        normal = normal/(np.linalg.norm(normal))
        if not np.isnan(np.sum(normal)):
            normal_map[xT] = normal
        i = i+1
'''
#print(normal)
#alb = Image.fromarray(L)

depth = Image.fromarray(normal)
rgb = Image.fromarray(normalrgb,'RGB')
nmap1 = Image.fromarray(normalsx)
nmap2 = Image.fromarray(normalsy)
nmap3 = Image.fromarray(normalsz)
nmap4 = Image.fromarray(normal)

#nmap1.show()
#nmap2.show()
#nmap3.show()
depth.show()
rgb.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(range(0,500), range(0,640))
ax.contour3D(X, Y, depth, 500,cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

ax.view_init(90, 70)
plt.show()






