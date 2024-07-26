# Course structure

- supervised learning

- unsupervised learning

- neutral network & LSTM

- time series

# Optional content

![z5664908701455_ce0140bb6e767ece974da5c5af202e25](https://github.com/user-attachments/assets/393f0795-e9b2-44b8-94dd-195524184eff)

1) Kết nối giao diện người dùng với AI và publish lên thành web-app responsive

2) Nhằm mục đích tăng tốc tính toán để mô hình có thể học trên tập dữ liệu lớn

Có 2 trường phái xử lý dữ liệu 

- xử lý bằng big data (tập hợp sức mạnh của nhiều phần cứng lại)

- tăng tốc tính toán (sử dụng CPU)

3) Ở industry practice thì qtam đến yếu tố interactive and reliability

- phải giái thích đc vì sao mô hình lại đưa ra quyết định như vậy. dựa trên thông tin và gợi ý gì.

- ví dụ: fraud detection -> ví dụ khách hàng sử dụng mô hình không đc như ý gây ra những thiệt hại. AI không thể tự chịu trách nhiệm. Nhiệm vụ của chúng ta là đưa ra lời gt cho khách hàng

4) Hệ thống khuyến nghị sản phẩm, tự động đề xuất cho kh những sp phù hợp

5) build chatbot = chatgpt

# Final exam

- Case study: có 1 tập data set liên quan tới vấn đề có thật

- Portfolio: small story telling (ex: tập dataset về khách hàng, sau khi phân tích thì kết quả phân tích phải đc sử dụng để phân loại khách hàng)

- có điểm cộng khi tham gia đầy đủ vào các lớp học. có điểm engagement

# Achievement

# Buổi học 1: Tổng quan Machine Learning (24/07/2024)

## 1. Giới thiệu 

- Machine learning và AI khác nhau ntn?

AI là trí tuệ nhân tạo, là hệ thống máy móc có suy nghĩ như con ngừoi. Machine learning là máy móc để tạo ra AI

2 hướng tiếp cận: instruction (traditional) và data driven (đây là hướng tiếp cận của machine learning)

- Vì sao để đào tạo AI có nhiều phương pháp nhưng vào ngày nay thì machine learning lại phổ biến hơn?

PP cổ điển: Khi chúng ta triển khai thực tế thì bảng hướng dẫn không thể bao gồm tất cả các tình huống xảy ra trong thực tế -> khi AI gặp tình huống k có trong tập hướng dẫn này thì sẽ bị đứng im. (Những pp cổ điển có tính tổng quát hoá generalization không cao 

PP hiện tại: data nhiều -> mô hình phân tích càng chính xác-> phân tích đc nhiều tình huống mà chúng ta không nghĩ ra 

- Phầm lớn các bài toán trong ML tiếp cận theo hướng thống kê xác xuất hoặc hướng đại số tuyến tính

- Deep learning là một field ngách của ML sử dụng mô hình kiến trúc nơ rong network đa lớp (vd: deep fake, chat gpt,..)

### Machine learning là ...

![Ảnh màn hình 2024-07-26 lúc 17 19 33](https://github.com/user-attachments/assets/3613d08b-7497-4929-8f48-81c08959aabf)

 hidden pattern là những thông tin ẩn mà chúng ta thu thập đc trong qtrinh ML

### Một số ứng dụng của ML

<img width="844" alt="Ảnh màn hình 2024-07-26 lúc 17 24 14" src="https://github.com/user-attachments/assets/2cc25be9-5aed-4404-b01e-689d1e4a2a2e">

<img width="837" alt="Ảnh màn hình 2024-07-26 lúc 17 24 19" src="https://github.com/user-attachments/assets/15aae597-ec5e-4783-8752-6bd4be461478">

<img width="1086" alt="Ảnh màn hình 2024-07-26 lúc 17 24 24" src="https://github.com/user-attachments/assets/75117950-5a29-492a-a4cc-471c908318b0">

## 2. Phân loại 

- supervised learning: học có giám sát

 thường thì supervised learning sẽ đưa ra kết quả chính xác hơn 

 VD: Regression(đầu ra y là biến continous không giới hạn), Classification(đầu ra y là biến categorical giới hạn)

- unsupervised learning: học không giám sát

 VD: clustering(

- semi-supervised learning: học bán giám sát

- reinforcement learning

<img width="1118" alt="Ảnh màn hình 2024-07-26 lúc 17 35 20" src="https://github.com/user-attachments/assets/e6be5c25-1225-4116-b5a6-6b394613ad15">

### When Do We Use Machine Learning?

ML is used when:

• Human expertise does not exist (navigating on Mars)

• Humans can’t explain their expertise (speech recognition) 

ML có thể đưa ra được lời giải thích cho những vấn đề con người không thể giải thích 

• Models must be customized (personalized medicine)

• Models are based on huge amounts of data (genomics)


## 3. Thách thức của Machine Learning 

## 4. Thuật ngữ 

## 5. Làm việc với dự án Machine Learning 

## 6. Môi trường làm việc

# Buổi học 2: Supervised Learning- Naïve Bayes (26/07/2024)

https://www.kaggle.com/code/ericle3121/naive-bayes-tutorial-from-beginning-tunning?scriptVersionId=189701710

https://monkeylearn.com/blog/what-is-tf-idf/?authuser=0

### Probability 

- p(A): xác xuất diễn ra sự kiện A 

- p(A/B): xác xuất có điều kiện A xảy ra, giả sử rằng B đã done (conditional probility)

- p(A,B): xác xuất để A và B xảy ra đồng thời 

## 1. Classification

- Classification là Supervised
  
### Phân loại 

## 2. Giới thiệu Naïve Bayes 

- hoạt động khi biến output là biến categorical

- trong trường hợp biến output là continous thì phỉa thay bằng

<img width="1059" alt="Ảnh màn hình 2024-07-26 lúc 19 31 06" src="https://github.com/user-attachments/assets/8c7f1640-61f6-4777-bab7-9717c30f499d">

## 3. Thuật toán

### 1 số ứng dụng phổ biến có thể kể đến như 

- text classification (vd: phân loại bình luận, lọc thư rác,...)

- Bayesian Classification

- handwritten digit recognition

![Ảnh màn hình 2024-07-26 lúc 18 38 31](https://github.com/user-attachments/assets/eebcb6a0-490d-4627-a1a6-f9cded673e45)

### The Naïve Bayes Assumption

<img width="983" alt="Ảnh màn hình 2024-07-26 lúc 19 38 17" src="https://github.com/user-attachments/assets/b46c9bd9-d546-42df-a7c5-6aee659d1c13">

- Khuyết điểm:

Trong thực tế, giả định này thường không đúng vì các đặc trưng có thể có mối quan hệ phụ thuộc lẫn nhau.

Chỉ cần 1 cái bằng không thì kết quả tổng thể sẽ bị sai dù đã có cơ chế 

Không xem xét và không coi trọng trật tự các từ trong câu (do thuật toán có tính giao hoán)

## 4. Ưu/khuyết điểm

- Khuyết điểm:

khi chúng ta dự đoán 1 mẫu chưa từng tồn tại trong data set thì sẽ ra kq = 0.

<img width="1013" alt="Ảnh màn hình 2024-07-26 lúc 19 22 31" src="https://github.com/user-attachments/assets/fe056f35-937a-440b-ad3c-bbb9ae7f8e76">


## 5. Xây dựng Naive Bayes sử dụng sklearn
