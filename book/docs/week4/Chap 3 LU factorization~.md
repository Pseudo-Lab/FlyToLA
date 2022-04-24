# #Chap.3: LU factorization ~ 끝까지

날짜: January 25, 2022
발표자: 박민호
자료: https://angeloyeo.github.io/2021/06/16/LU_decomposition.html, https://courses.engr.illinois.edu/cs554/fa2013/notes/06_lu_8up.pdf, [shabat2016](./asset/shabat2016.pdf)

## Objectives

- LU Decomposition은 왜 쓰는가?
    - 계산의 편리함을 위해 사용
- LU Decomposition 사용 시 유의해야 할 점
    - 안정성
    - 계산 상의 이득을 얻을 수 있는 세팅
- LU Decomposition의 Speed Up 방법
    - parallel execution 사용 방법 (usually CPU)
    - randomized LU Decomposition (GPU)

---

## LU Decomposition

### why?

- Matrix를 Decomposition 하는 이유
    1. **계산의 편리함 (LU Decomposition)**
    2. 분석적 용이성 (SVD, PCA ...)

### Gaussian Elimination

- LU Decomposition은 Gaussian elimination에 기초를 두고 있다.

![Gaussian elimination (출처 : [https://en.wikipedia.org/wiki/Gaussian_elimination](https://en.wikipedia.org/wiki/Gaussian_elimination))](asset/Untitled.png)

Gaussian elimination (출처 : [https://en.wikipedia.org/wiki/Gaussian_elimination](https://en.wikipedia.org/wiki/Gaussian_elimination))

- 위 table에서 3행 까지를 보면 upper triangular matrix인 것을 알 수 있다. (LU에서의 U)
- 그럼 lower triangular matrix를 구하는 방법은? → Gaussian elimination 과정을 matrix의 곱으로 나타내면 됨!
- 위 과정을 마지막 미지수부터 처음 미지수로 계산해 나간다고 하여 back substitution이라고 부른다.

### Back substitution

- 기존 Augmented matrix에서

$$
A = \begin{bmatrix}
  2 & 1 & -1 & |& 8  \\
  -3 & -1 & 2 &|& -11  \\
 -2 & 1 & 2 &|& -3
\end{bmatrix}
$$

- 2, 3행의 row operation을 행렬 곱으로 나타내면 다음과 같다. ([기본행렬](https://angeloyeo.github.io/2021/06/15/elementary_square_matrices.html) 참고)

$$
\begin{bmatrix}
  1 & 0 & 0   \\
  0 & 1 & 0  \\
  0 & -4 & 1 
\end{bmatrix}
\begin{bmatrix}
  1 & 0 & 0   \\
  3/2 & 1 & 0  \\
 1 & 0 & 1 
\end{bmatrix}
\begin{bmatrix}
  2 & 1 & -1 & |& 8  \\
  3 & -1 & 2 &|& -11  \\
 -2 & 1 & 2 &|& -3
\end{bmatrix}
= \begin{bmatrix}
  2 & 1 & -1 & |& 8  \\
  0 & 1/2 & 1/2 &|& 1  \\
 0 & 0 & -1 &|& 1
\end{bmatrix}
$$

### 기본 행렬의 역행렬 곱하기 → LU 분해

- 앞의 두 기본 행렬을 각각 $e_1, e_2$  라고 한다면 (기본 행렬의 역행렬 참고)

$$
e_1^{-1} = 
\begin{bmatrix}
  1 & 0 & 0   \\
  0 & 1 & 0  \\
  0 & 4 & 1 
\end{bmatrix}
$$

$$
e_2^{-1} = 
\begin{bmatrix}
  1 & 0 & 0   \\
  -3/2 & 1 & 0  \\
  -1 & 0 & 1 
\end{bmatrix}
$$

- back substitution에서 행렬만을 생각했을 때,

$$
A = \begin{bmatrix}
  2 & 1 & -1   \\
  3 & -1 & 2   \\
 -2 & 1 & 2 
\end{bmatrix}=
e_2^{-1}e_1^{-1}U=
\begin{bmatrix}
  1 & 0 & 0   \\
  -3/2 & 1 & 0  \\
  -1 & 0 & 1 
\end{bmatrix}
\begin{bmatrix}
  1 & 0 & 0   \\
  0 & 1 & 0  \\
  0 & 4 & 1 
\end{bmatrix}
\begin{bmatrix}
  2 & 1 & -1  \\
  0 & 1/2 & 1/2   \\
 0 & 0 & -1
\end{bmatrix}=
\begin{bmatrix}
  1 & 0 & 0   \\
  -3/2 & 1 & 0  \\
  -1 & 4 & 1 
\end{bmatrix}
\begin{bmatrix}
  2 & 1 & -1  \\
  0 & 1/2 & 1/2   \\
 0 & 0 & -1
\end{bmatrix}
$$

- 따라서, $L = e_2^{-1}e_1^{-1}$ 이다.

### back substitution과 기본 행렬 역행렬을 동시에 처리

```python
def LU(A):
    U = np.copy(A)
    m, n = A.shape
    L = np.eye(n)
    for k in range(n-1):
        for j in range(k+1,n):
            L[j,k] = U[j,k]/U[k,k]
            print("L")
			      print(L)
			      U[j,k:n] -= L[j,k] * U[k,k:n]
			      print("U")
			      print(U)
    return L, U
```

- 계산 복잡도
    - for, for, U[k,k:n](평균 적으로 2/n) → n^3
    - numpy의 U[k,k:n]이 빠르게 되는 이유 → 일이 sequential하게 일어나지 않아도 되는 부분으로vectorization 연산으로 빠르게 가능(SIMD)
- 위 코드를 실행했을 때,

```python
print(A)
>>>
[[ 2.  1. -1.]
 [-3. -1.  2.]
 [-2.  1.  2.]]

LU(A)
>>>
L
[[ 1.   0.   0. ]
 [-1.5  1.   0. ]
 [ 0.   0.   1. ]]
U
[[ 2.   1.  -1. ]
 [ 0.   0.5  0.5]
 [-2.   1.   2. ]]
L
[[ 1.   0.   0. ]
 [-1.5  1.   0. ]
 [-1.   0.   1. ]]
U
[[ 2.   1.  -1. ]
 [ 0.   0.5  0.5]
 [ 0.   2.   1. ]]
L
[[ 1.   0.   0. ]
 [-1.5  1.   0. ]
 [-1.   4.   1. ]]
U
[[ 2.   1.  -1. ]
 [ 0.   0.5  0.5]
 [ 0.   0.  -1. ]]

```

### LU decomposition의 계산 복잡도 및 공간 복잡도

- gauss elimination의 계산 복잡도는 $O(2 \cdot\frac{1}{3}n^3)$
- 메모리 사용량은, 위의 코드에서 `U=np.copy(A)` 부분을 U를 A를 바로 inplace하는 방식으로 메모리 절감가능, 그리고 L의 경우 diagonal성분이 모두 1이기 때분에 메모리에 따로 저장을 할 필요 없이, lower triangle 부분의 값들만 저장하여 계산 가능
    - 반면, inverse matrix를 사용하여 행렬식 계산시에는 원 행렬만큼의 메모리가 필요
    - 원 행렬이 sparse할 경우, L,U도 sparse 하지만, 역행렬은 dense하다.

### Stability of LU

- 원 행렬의 대각 성분이 0에 가까울수록 LU decomposition은 unstable 해진다.
    - `L[j,k] = U[j,k]/U[k,k]` 에서 U[k,k]가 0에 가까우면 L[j,k]의 numerical error가 커짐
- 따라서 대각 행렬의 성분이 적절하도록 **pivoting**이 필요

### LU factorization with Pivoting

- 행렬에서 열의 순서를 바꾸는 것을 **pivoting**이라고 한다.
- 이는 [기본행렬](https://angeloyeo.github.io/2021/06/15/elementary_square_matrices.html)에서 Row switching 참고
- pivot 이후 LU 분해하는 것을 **PLU 분해**라고 함

### scipy.linalg.solve vs scipy.linalg.lu_solve

- scipy.linalg.solve
    - `scipy.linalg.solve(A, b)`
    - $O(n^3)$
    - reciprocal pivot growth factor라는 것을 이용하여, 일반적인 lu factorization보다 더 complex한 과정을 거쳐 linear system을 solve
    - 일반적인 lu factorization을 이용한 solve (lu_solve)보다 더 stable 함
- scipy.linalg.lu_solve
    - `scipy.linalg.lu_solve(scipy.linalg.lu_factor(A), b)`
    - scipy.linalg.lu_solve : $O(n^2)$
        - 행렬 크기만큼 계산함으로
    - scipy.linalg.lu_factor : $O(n^3)$
    - A가 고정이고, b가 여러개인 상황에서는 lu_factor를 한번만 계산하고, lu_solve를 여러번 계산하면 계산상의 이득이 존재
    

### LU decomposition Speed Up!

- [Parallel LU decomposition](https://courses.engr.illinois.edu/cs554/fa2013/notes/06_lu_8up.pdf) - parallel execution (usually at CPU)
- [Randomized LU decomposition](https://www.sciencedirect.com/science/article/abs/pii/S1063520316300069) - using GPU

## ~~History of Gaussian Elimination~~

## Block Matrix Multiplication

![Untitled](asset/Untitled%201.png)

- Big O notation : $O(n^3)$ - 기본 matrix multiplication이랑 같다.
- 하지만 계산상 몇가지 이점이 있음

### 계산 이점

1. 특정 행렬을 이용한 계산 감소
    
    $$
    \begin{equation*} A = \left[ \begin{array}{rr|rrr} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ \hline 2 & -1 & 4 & 2 & 1 \\ 3 & 1 & -1 & 7 & 5 \end{array} \right] = \left[ \begin{array}{cc} I_{2} & O_{23} \\ P & Q  \end{array} \right] \end{equation*}
    $$
    

$$
\begin{equation*} B = \left[ \begin{array}{rr} 4 & -2 \\ 5 & 6 \\ \hline 7 & 3 \\ -1 & 0 \\ 1 & 6 \end{array} \right] = \left[ \begin{array}{c} X \\ Y \end{array} \right] \end{equation*}
$$

$$
\begin{equation*} AB = \left[ \begin{array}{cc} I & O \\ P & Q \end{array} \right] \left[ \begin{array}{c} X \\ Y \end{array} \right] = \left[ \begin{array}{c} IX + OY \\ PX + QY \end{array} \right] = \left[ \begin{array}{c} X \\ PX + QY \end{array} \right] = \left[ \begin{array}{rr} 4 & -2 \\ 5 & 6 \\ \hline 30 & 8 \\ 8 & 27 \end{array} \right] \end{equation*}
$$

1. Strassen  algorithm
    1. [https://ko.wikipedia.org/wiki/슈트라센_알고리즘](https://ko.wikipedia.org/wiki/%EC%8A%88%ED%8A%B8%EB%9D%BC%EC%84%BC_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)
    2. $O(n^{2.807})$
    
2. **SIMD operation**
    1. [https://stackoverflow.com/questions/44944367/are-numpys-basic-operations-vectorized-i-e-do-they-use-simd-operations](https://stackoverflow.com/questions/44944367/are-numpys-basic-operations-vectorized-i-e-do-they-use-simd-operations)
    2. [https://ko.m.wikipedia.org/wiki/SIMD](https://ko.m.wikipedia.org/wiki/SIMD)
        1. 연속된 주소값의 데이터를 계산
    

---

## Additional

### PCP (Principal Component Pursuit)

- [https://nbviewer.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3. Background Removal with Robust PCA.ipynb#Robust-PCA-(via-Primary-Component-Pursuit)](https://nbviewer.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb#Robust-PCA-(via-Primary-Component-Pursuit))
- augmented Lagrange method : [https://wikidocs.net/24019](https://wikidocs.net/24019)
- [https://arxiv.org/pdf/0912.3599.pdf](https://arxiv.org/pdf/0912.3599.pdf) 29 page
- [https://arxiv.org/pdf/1009.5055.pdf](https://arxiv.org/pdf/1009.5055.pdf) algorithm3, 5

### 질의응답+추가설명

- LU factorization에서 L,U의 의미는? → 정성적 의미가 있다로 해석할 필요는 없을듯
    - L: gaussian의 elimination 에 사용된 operation의 inverse 들의 곱
    - U: 풀고자하는 꼴로 만들어진게 U
- gaussian elimination 과정에서 3*1/2 * n^3이 어케 되는건지
- LU factor를 scipy로 하면, L,U말고도 P(piv)도 함께 리턴됨
    - [2,2,3,3] : 0부터 시작하면서 0번재는 2랑 바꾸고, 첫번재는 0이된 2로 바꾸는거 ..
    - [0,1,2,3] :
- (참고자료의 논문) A → LU 근사하는 것을 빠르게 하고 싶은 randomized LU decomposition.
    - matrix multiplication 이 n^3 아닌가? → 우리가 GPU를 사용하고 잇고, sequential한 dependency가 발생하지 않음으로 GPU 활용 가능. 알고리즘에서 Lu decomposition을 사용하긴 하지만 크기를 줄여서 decomposition에 연관되는 n의 사이즈를 줄여서 연산 줄인 것으로. + n 을 줄엿기 때문에 inverse matrix 구하는 것도 이득