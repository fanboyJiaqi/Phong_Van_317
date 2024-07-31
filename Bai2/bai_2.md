Bài 2
# Công thức toán học Triplet Loss:

$$
L(A, P, N) = \max \left(0, \|f(A) - f(P)\|_2^2 - \|f(A) - f(N)\|_2^2 + \alpha \right)
$$

Trong đó:
- \( A \) là mẫu anchor.
- \( P \) là mẫu positive.
- \( N \) là mẫu negative.
- \( f \) là hàm biểu diễn đặc trưng.
- \( \| \cdot \|_2 \) là chuẩn Euclidean.
- \( \alpha \) là khoảng cách biên (margin).

![alt text](image.png)   ![alt text](image-1.png)

Mục tiêu: tối thiểu hóa khoảng cách giữa 2 ảnh khi chúng là negative và tối đa hóa khoảng cách khi chúng là positive. Lựa bộ ba ảnh:
	- Ảnh Anchor và Positives khác nhau nhất: cần lựa chọn để d(A,P) lớn. Điều này tương tự như lựa chọn một ảnh của mình hồi nhỏ vs hiện tại để thuật toán khó học hơn. Nhưng học được sẽ thông minh hơn.
	- Ảnh Anchor và Negatives giống nhau nhất: cần lựa chọn để d(A,N) nhỏ.

# Công thức toán học Triplet Loss với nhiều mẫu Positive và Negatives

Với \( n \) mẫu positive và \( m \) mẫu negative, công thức được mở rộng thành:

$$
L = \sum_{i=1}^{n} \sum_{j=1}^{m} \max \left(0, \|f(A) - f(P_i)\|_2^2 - \|f(A) - f(N_j)\|_2^2 + \alpha \right)
$$

Trong đó:
- \( P_i \) là các mẫu positive (có \( n \) mẫu).
- \( N_j \) là các mẫu negative (có \( m \) mẫu).