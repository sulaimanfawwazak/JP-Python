from charminal import *
########## SETS DAN VECTOR ##########
#####################################
print(f'{COLOR_GREEN}\n########## SETS DAN VECTOR ##########{RESET}')

########## 01: JALANKAN! Buat row vector dan column vector, dan tampilkan vector shape nya. ##########
print(f'01')
import numpy as np

vector_row = np.array([[1, -5, 3, 2, 4]])
vector_column = np.array([[1],
                          [2],
                          [3],
                          [4]])

print(vector_row.shape)
print(vector_column.shape)

########## 02: JALANKAN! Transpose vektor baris yang kita definisikan di atas menjadi vektor kolom dan hitung norm LI, L2, dan Lo darinya. Verifikasi bahwa norm Loo vektor setara dengan nilai maksimum elemen dalam vektor ##########
print(f'')
print(f'02')
from numpy.linalg import norm

new_vector = vector_row.T
print(new_vector)

norm_1 = norm(new_vector, 1)
norm_2 = norm(new_vector, 2)
norm_inf = norm(new_vector, np.inf)

print(f'L_1 is: {norm_1:.1f}')
print(f'L_2 is: {norm_2:.1f}')
print(f'L_inf is: {norm_inf:.1f}')

########## 03: JALANKAN: Tunjukkan bahwa a(v+w)=av+aw (yaitu, perkalian skalar dari suatu vektor terdistribusi ke seluruh penjumlahan vektor). ##########
print(f'')
print(f'03')
a = 3
v = np.array([[1, 2, 3, 4, 5]])
w = np.array([[10, 20, 30, 40, 50]])

v_plus_w = v+w

print(f'v_plus_w: {v_plus_w}')

a_times_v_plus_w = a * v_plus_w
print(f'a_times_v_plus_w: {a_times_v_plus_w}')

a_times_v = a * v
a_times_w = a * w

print(f'a_times_v: {a_times_v}')
print(f'a_times_w: {a_times_w}')

a_times_v_plus_a_times_w = a_times_v + a_times_w
print(f'a_times_v_plus_a_times_w: {a_times_v_plus_a_times_w}')

if np.array_equal(a_times_v_plus_w, a_times_v_plus_a_times_w):
  print(f'VECTORS ARE THE SAME')
else:
  print(f'VECTORS ARE NOT THE SAME!')

########## 04: JALANKAN! Hitung sudut antara vektor v=[11,9,3) dan w=[2,5,12). ##########
print(f'')
print(f'04')
from numpy import arccos, dot

v = np.array([[10, 9, 3]])
w = np.array([[2, 5, 12]])

theta = \
  arccos(dot(v, w.T) / (norm(v) * norm(w)))

print(theta)

########## 05: JALANKAN! Diberikan vektor v=[0,2,0] dan w=[3,0,0|, gunakan fungsi silang Numpy untuk menghitung perkalian silang dari v dan w. ##########
print(f'')
v = np.array([[0, 2, 0]])
w = np.array([[3, 0, 0]])

print(np.cross(v, w))

########## 06: JALANKAN! Diberi vektor baris v=[0,3,2], w=[4,1,1], dan u=[0,—2,0], tulis vektor X-(-8,—1,4) sebagai kombinasi linear dari v, w, dan u. ##########
print(f'')
v = np.array([[1, 1, 0]])
w = np.array([[1, 0, 0]])
u = np.array([[0, 0, 1]])

matrix = np.vstack([v, w, u])

rank = np.linalg.matrix_rank(matrix)

print(f'Rank: {rank}')

is_independent = rank == matrix.shape[0] # Checks if rank equals to the number of rows

print(f'Are the vectors linearly independent? {is_independent}')
v = np.array([[0, 3, 2]])
w = np.array([[4, 1, 1]])
u = np.array([[0, -2, 0]])
x = 3*v - 2*w + 4*u
print(x)

########## 07: JALANKAN! Tentukan dengan inspeksi apakah himpunan vektor berikut linearly independent ##########
print(f'')

v = np.array([[1, 1, 0]])
w = np.array([[1, 0, 0]])
u = np.array([[0, 0, 1]])

matrix = np.vstack([v, w, u])

rank = np.linalg.matrix_rank(matrix)

print(f'Rank: {rank}')

is_independent = rank == matrix.shape[0] # Checks if rank equals to the number of rows

print(f'Are the vectors linearly independent? {is_independent}')

########## MATRIX ##########
############################
print(f'{COLOR_GREEN}\n########## MATRIX ##########{RESET}')

########## 01: JALANKAN! Jika matriks P=[[1,7],[2,3],[5,0]] dan Q=[[2,6,3,1],[1,2,3,4]]. Hitung hasil kali matriks P dan O. Tunjukkan bahwa hasil kali O dan P akan menghasilkan kesalahan. ##########
print(f'01')
P = np.array([[1, 7], [2, 3], [5, 0]])
Q = np.array([[2, 6, 3, 1], [1, 2, 3, 4]])

print(f'P:\n{P}')
print(f'Q:\n{Q}')

# Ini aku tambahin try-except supaya bisa di-run
try: 
  print(f'P x Q')
  print(np.dot(P, Q))
except Exception as e:
  print(e)

try:
  print(f'Q x P')
  print(np.dot(Q, P))
except Exception as e:
  print(e)

########## 02: JALANKAN! Gunakan Python untuk mencari determinan matriks M=[[0,2,1,3],[3,2,8,1],[1,0,0,3],[0,3,2,1]]. Gunakan fungsi np.eye untuk menghasilkan matriks identitas 4x4, I. Kalikan M dengan I untuk menunjukkan bahwa hasilnya adalah M.
print(f'')
print(f'02')

from numpy.linalg import det

M = np.array([[0, 2, 1, 3],
              [3, 2, 8, 1],
              [1, 0, 0, 3],
              [0, 3, 2, 1]])

print(f'M:\n{M}')

print(f'Determinant: {det(M):.1f}')

I = np.eye(4)
print(f'I\n:{I}')

print(f'M*I:\n{np.dot(M, I)}')

########## 03: JALANKAN! Matriks M (dalam contoh sebelumnya) memiliki: determinan bukan nol. Hitung invers dari M. Tunjukkan bahwa matriks P-(1O,1,01,10,0,01,11,0,1)) memiliki nilai determinan 0 dan karenanya tidak memiliki invers. |
print(f'')
print(f'03')

from numpy.linalg import inv

print(f'Inv M:\n', inv(M))

P = np.array([[0, 1, 0],
              [0, 0, 0],
              [1, 0, 1]])

print(f'det(P):\n', det(P))

########## 04: JALANKAN! Matniks A=[[1,1,0],[0,1,0],[1,0,1]], hitung bilangan kondisi dan rank untuk matriks ini. Jika y=[[1],[2],[1]], dapatkan matriks yang diperbesar [A, y]. | ##########
print(f'')
print(f'04')

from numpy.linalg import cond, matrix_rank

A = np.array([[1, 1, 0],
              [0, 1, 0],
              [1, 0, 1]])

print(f'Condition number:\n', cond(A))
print(f'Rank:\n', matrix_rank(A))

y = np.array([[1], [2], [1]])

A_y = np.concatenate((A, y), axis=1)
print(f'Augmented matrix:\n{A_y}')

########## SOLUSI SISTEM PERSAMAAN LINEAR ##########
####################################################
print(f'{COLOR_GREEN}\n########## SOLUSI SISTEM PERSAMAAN LINEAR ##########{RESET}')

########## 01: JALANKAN:! Pastikan matriks L dan U di bawah adalah dekomposisi LU dari matriks A. |: Kita akan melihat bahwa A—LU. Matriks A dibentuk dari persamaan berikut: |##########
print(f'01')

l = np.array([[4, 3, -5],
              [0, -2.5, 2.5],
              [0, 0, 60]])

u = np.array([[1, 0, 0],
              [-0.5, 10, 0],
              [2, -0.8, 1]])

print(f'LU = \n', np.dot(l, u))

########## 02: JALANKAN! Selesaikan sistem persamaan linier berikut menggunakan metode Gauss- Seidel, gunakan ambang batas yang telah ditentukan sebelumnya e-0,01. Ingatlah untuk memeriksa apakah kondisi konvergen terpenuhi atau tidak ##########
print(f'')
print(f'02')

a = [[8, 3, -3], [-2, -8, 5], [3, 5, 10]]

# Find diagonal coefficients
diag = np.diag(np.abs(a))

# Find row sum without diagonal
off_diag = np.sum(np.abs(a), axis=1) - diag

if np.all(diag > off_diag):
  print(f'Matrix is diagonally dominant')
else:
  print(f'NOT diagonally dominant')

x1 = 0
x2 = 0
x3 = 0
epsilon = 0.01 
converged = False

x_old = np.array([x1, x2, x3])

print("Iteration results")
print("k,\tx1,\tx2,\tx3")
for k in range(1, 50):
  x1 = (14 - 3*x2 + 3*x3)/8
  x2 = (5 + 2*x1 -5*x3)/(-8)
  x3 - (-8- 3*x1 - 5*x2)/(-5)
  x - np.array([x1, x2, x3])

  # check if it is smaller than threshold
  dx = np.sqrt(np.dot(x-x_old, (x-x_old).T)) # Ini di modul ga ada .T nya, tambahin sendiri biar bisa di dot product

  print(f'{k:d}, {x1:.4f}, {x2:.4f}, {x3:.4f}')
  if dx < epsilon:
    converged = True
    print('Converged!')
    break

  # assign the latest x value to the old value
  x_old = x

if not converged:
  print("Not converge, increase the # of iterations")

########## MENYELESAIKAN SISTEM PERSAMAAN LINEAR DENGAN PYTHON ##########
#########################################################################
print(f'{COLOR_GREEN}\n########## MENYELESAIKAN SISTEM PERSAMAAN LINEAR DENGAN PYTHON ##########{RESET}')

########## 01: JALANKAN! Gunakan numpy.linalg.solve untuk menyelesaikan persamaan berikut. ##########
print(f'01')

A = np.array([[4, 3, -5],
              [-2, -4, 5],
              [8, 8, 0]])
y = np.array([2, 5, -3])

x = np.linalg.solve(A, y)

print(f'x:\n', x)

########## 02: JALANKAN! Coba selesaikan persamaan di atas menggunakan pendekatan inversi matriks ##########
print(f'')
print(f'02')

A_inv = inv = np.linalg.inv(A)
x = np.dot(A_inv, y)

print(f'x:\n', x)

########## 03: JALANKAN! Dapatkan L dan U untuk matriks A di atas. ##########
print(f'')
print(f'03')

from scipy.linalg import lu

P, L, U = lu(A)

print(f'P:\n', P)
print(f'P:\n', L)
print(f'P:\n', U)
print(f'LU:\n', np.dot(L, U))

########## 04: JALANKAN! Kalikan P dan A dan lihat apa pengaruh matriks permutasi pada A. ##########
print(f'')
print(f'04')

print('P x A =\n', np.dot(P, A))

########## TUGAS ##########
###########################
print(f'{COLOR_GREEN}\n########## TUGAS ##########{RESET}')

########## Selesaikan persamaan berikut dengan menggunakan numpy.linalg.solve dan analisislah: ##########
A = np.array([[3, -1, 4],
              [17, 2, 1],
              [1, 12, -7]])

y = np.array([2, 14, 54])

x = np.linalg.solve(A, y)

print(f'x:\n', x)


