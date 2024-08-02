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

<img width="744" alt="Ảnh màn hình 2024-07-26 lúc 17 24 14" src="https://github.com/user-attachments/assets/2cc25be9-5aed-4404-b01e-689d1e4a2a2e">

<img width="737" alt="Ảnh màn hình 2024-07-26 lúc 17 24 19" src="https://github.com/user-attachments/assets/15aae597-ec5e-4783-8752-6bd4be461478">

<img width="986" alt="Ảnh màn hình 2024-07-26 lúc 17 24 24" src="https://github.com/user-attachments/assets/75117950-5a29-492a-a4cc-471c908318b0">

## 2. Phân loại 

**1. supervised learning: học có giám sát**

 – Given: training data + desired outputs (labels)

 thường thì supervised learning sẽ đưa ra kết quả chính xác hơn 

- Thường sẽ có 2 dạng thuật toán trong học có giám sát là Regression(đầu ra y là biến continous có thể có vô số giá trị), Classification(đầu ra y là biến categorical hữu hạn các giá trị có thể có)

- Regression là bài toán dự đoán 1 giá trị 

<img width="855" alt="Ảnh màn hình 2024-07-28 lúc 08 27 50" src="https://github.com/user-attachments/assets/ce0db7e0-58ad-4305-80a6-020f92497297">

VD về bài toán Regression: dự đoán giá nhà, dự đoán điểm của sinh viên,...

VD về bài toán  Classification: phân biệt 1 email là spam hay k spam, phân biệt review positive/negative

**2. unsupervised learning: học không giám sát**

– Given: training data (without desired outputs)

dữ liệu chưa đc đánh nhãn, dữ liệu mà chính chúng ta cũng chưa đưa ra lời giải thích dc (ví dụ như khám phá sao hoả 

 VD: sử dụng clustering(customer segmentation, market segmentation, social network analysis,..)

**3. semi-supervised learning: học bán giám sát**

– Given: training data + a few desired outputs

**4. reinforcement learning**

– Rewards from sequence of actions

<img width="1118" alt="Ảnh màn hình 2024-07-26 lúc 17 35 20" src="https://github.com/user-attachments/assets/e6be5c25-1225-4116-b5a6-6b394613ad15">

### When Do We Use Machine Learning?

ML is used when:

• Human expertise does not exist (navigating on Mars)

• Humans can’t explain their expertise (speech recognition) 

ML có thể đưa ra được lời giải thích cho những vấn đề con người không thể giải thích 

• Models must be customized (personalized medicine)

• Models are based on huge amounts of data (genomics)

### Designing a Learning System (Những thứ cần biết trước quá trình huấn luyện mô hình)

- Choose the training experience (có đầy đủ data)
 
- Choose exactly what is to be learned (xác định rõ ràng đầu ra cần phải là gì để xác định chọn mô hình gì)

![Ảnh màn hình 2024-07-28 lúc 08 50 43](https://github.com/user-attachments/assets/b95a57b0-c4d8-484b-8e77-c9fd97c9fa19)

### Training vs. Test Distribution

- Tập train và tập test phải có cùng 1 distribution thì mô hình mới có kq dự đoán tốt nhất được
  
## 3. Thách thức của Machine Learning 

- Không đủ số lượng dữ liệu đào tạo (training data)

- Dữ liệu đào tạo không đại diện (nonrepresentative)

sampling bias (xảy ra khi mẫu được chọn không đại diện cho tổng thể, dẫn đến các kết quả nghiên cứu hoặc dự đoán bị sai lệch), survival bias

- Dữ liệu có chất lượng kém (Poor-Quality Data)

missing data, irrelevant feature, duplicate 

- Thuộc tính không liên quan (Irrelevant Feature)

- Overfitting dữ liệu huấn luyện: train tốt test lỏm

 mô hình sensitive to noise, mức chênh lệch giữa dự đoán và thực tế rất lớn 

Nguyên nhân: Điều này xảy ra khi tập dữ liệu huấn luyện có nhiễu (noise), hay mô hình quá phức tạp, tức là có quá nhiều tham số so với số dữ liệu quan sát được (thể hiện). Chính nhiễu đã gây tác động xấu tới quá trình dự đoán của mô hình với dữ liệu kiểm tra.

Cách sửa: Giảm nhiểu trong dữ liệu đào tạo, đơn gỉan hoá mô hình

- Underfitting dữ liệu huấn luyện: train lỏm test lỏm

Nguyên nhân: Điều này xảy xa khi mô hình đang xây dựng quá đơn giản so với tập dữ liệu.

![Ảnh màn hình 2024-07-28 lúc 09 20 51](https://github.com/user-attachments/assets/efc89313-afdf-4a0f-b6e6-5457739cd174)

bias là độ lệch giữa dự đoán của mô hình machine learning đưa ra so với thực tế, variance là độ phân tán của dữ liệu 

### Trade-off

<img width="683" alt="Ảnh màn hình 2024-07-28 lúc 09 22 20" src="https://github.com/user-attachments/assets/38695567-e590-4f71-9cb3-1ed41b90a7d3">

bias và variance có tính trade off

## 4. Thuật ngữ 

- Sample (mẫu): row, record, example, instance, observation

<img width="633" alt="Ảnh màn hình 2024-07-28 lúc 09 23 36" src="https://github.com/user-attachments/assets/0e938029-4b80-41f4-8470-613fe867e72c">

- variable(biến): attribute, field, feature, column, dimension

- Data type (kiểu dữ liệu):

<img width="682" alt="Ảnh màn hình 2024-07-28 lúc 09 25 14" src="https://github.com/user-attachments/assets/e007eaf9-5c7d-43b1-bff6-2bf793d64277">

## 5. Làm việc với dự án Machine Learning 

<img width="655" alt="Ảnh màn hình 2024-07-28 lúc 09 25 43" src="https://github.com/user-attachments/assets/a7a5093e-c504-4e3e-8ef6-3df14424e453">

- Có khá nhiều nguồn có thể nhận dữ liệu (miễn phí/ trả phí) như:

UC Irvine Machine Learning Repository

Kaggle datasets

Amazon's AWS datasets

Wikipedia's list of Machine Learning datasets

## 6. Môi trường làm việc

Python, Scikit-learn

# Buổi học 2: Supervised Learning - Naïve Bayes (26/07/2024)

https://www.kaggle.com/code/ericle3121/naive-bayes-tutorial-from-beginning-tunning?scriptVersionId=189701710

https://monkeylearn.com/blog/what-is-tf-idf/?authuser=0

https://machinelearningcoban.com

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

## 4. Ưu/khuyết điểm

- Khuyết điểm:

Trong thực tế, giả định này thường không đúng vì các đặc trưng có thể có mối quan hệ phụ thuộc lẫn nhau.

Chỉ cần 1 cái bằng không thì kết quả tổng thể sẽ bị sai dù đã có cơ chế Laplace smoothing 

Không xem xét và không coi trọng trật tự các từ trong câu (do thuật toán có tính giao hoán)

khi chúng ta dự đoán 1 mẫu chưa từng tồn tại trong data set thì sẽ ra kq = 0. Chúng ta sẽ cố gắng khắc phục bằng cơ chế Laplace smoothing 

<img width="1013" alt="Ảnh màn hình 2024-07-26 lúc 19 22 31" src="https://github.com/user-attachments/assets/fe056f35-937a-440b-ad3c-bbb9ae7f8e76">

## 5. Xây dựng Naive Bayes sử dụng sklearn

**Bước 1: EDA**

- data, bài toán dễ hay khó

**Bước 2: Huấn luận mô hình trên tập trainning**

**Bước 3: Test**

### Naive Bayes Tutorial (Naive Bayes Tutorial: From Beginning -> Tunning)

- Xem phân phối nhãn .valuecounts -> có thể visualize thấy đc dữ liệu balanced/imbalanced

Nếu imbalanced thì phải tìm cách khắc phục vì sẽ gây ra model bias 

- xem xét dữ liệu trùng .duplicated

trong trường hợp bị duplicated , xem xét thêm nhãn của nó có consistant với nhau không, nếu chúng consistant với nhau thì nên giữ lại

- xem statistic của từng mẫu

- showing word cloud

- import nltk là package xử lý ngôn ngữ tự nhiên

- Chuyển dữ liệu từ text thành bảng số để mô hình có thể đọc và hiểu bằng CountVectorizer

**Lưu ý**

ở tập train dùng fit transfrom , tập test dùng transform 

nếu dùng fit trên tập test thì sẽ bị trường hợp bias và data leaked

- from sklearn.naive_bayes import MultinomialNB

- from sklearn.feature_extraction.text import TfidfVectorizer (đổi từ CountVectorizer -> TfidfVectorizer)

- kỹ thuật TF-IDF (TF-IDF cao có nghĩa là 1 từ xuất hiện nhiều lần trong văn bản hiện tại, và xuất hiện rất ít trong những văn bản khác -> đó thường là từ đặc trưng của chuyên ngành đó)

- 1 từ có TF-IDF càng cao thì càng là từ quan trọng và đặc trưng

Giả sử trong các review tích cực, từ "good" và "excellent" xuất hiện thường xuyên, trong khi trong review tiêu cực, từ "bad" và "terrible" lại phổ biến hơn.

TF-IDF sẽ giúp mô hình Naive Bayes nhận ra rằng "good" và "excellent" quan trọng hơn "the" hoặc "is" trong việc phân loại review thành tích cực, và tương tự với "bad" và "terrible" trong review tiêu cực.


<img width="336" alt="Ảnh màn hình 2024-07-28 lúc 02 14 41" src="https://github.com/user-attachments/assets/912698f8-692f-495f-8eb1-358518cc86f4">

<img width="341" alt="Ảnh màn hình 2024-07-28 lúc 02 14 47" src="https://github.com/user-attachments/assets/2a94d71f-ef47-4194-91d9-f5c5b425a17b">

# Buổi học 3: Supervised Learning - K-Nearest Neighbors (28/07/2024)

https://www.gradio.app/guides/quickstart?authuser=1

K-Nearest Neighbors (KNN) là một thuật toán thuộc nhóm Supervised Learning được sử dụng cho classification và regression.

Với classification, ouput là class dựa trên KNN trong training data. Với regression, output là trung bình các giá trị của target variable dựa trên KNN trong training data

## 1. Giới thiệu

- Regression analysis là một dạng của kỹ thuật mô hình tiên đoán (predictive modelling technique)

- dự đoán ra output là giá trị thật thuộc R, có thể có vô số các giá trị có thể có

- cần data đã đc lable sẵn

- predict: dự đoán -> bao gồm tất cả các dự đoán

- forecast: dự báo, tiên đoán -> dự báo những gì chưa xảy ra

## 2.Thuật toán

- dựa trên quy tắc major voting (đi theo số đông)

**Q1: k = ? thì tốt?**

- giá trị k nhỏ sẽ gây ra overfitting. Vì chỉ dựa trên điểm gần nhất để đưa ra dự đoán -> chỉ cần một điểm dữ liệu lạ hoặc nhiễu là kết quả dự đoán có thể bị ảnh hưởng mạnh

- chọn k phải là số lẻ

- k != 1, k != N

- phải khảo sát k = (3,5,7,...) -> giá trị nào tốt trên tập validation thì chọn

- tìm optimal k: khảo sát trên 1 tập k và với mỗi giá trị k ghi lại train error và test error

![Ảnh màn hình 2024-08-02 lúc 00 32 30](https://github.com/user-attachments/assets/9646a0a3-5821-4adf-bc10-0d1f34b6f9a5)

**Q2: phép đo distance có đk nào không?**

- gặp KNN thì phải scale dữ liệu (biến continous) (để đảm bảo rằng tất cả các đặc trưng được đối xử công bằng trong quá trình tính toán khoảng cách, từ đó giúp thuật toán phân loại chính xác hơn. Nếu bỏ qua bước này, các đặc trưng với giá trị lớn hơn sẽ chi phối kết quả và làm cho mô hình kém hiệu quả.)

- biến categorical thì làm như sau
 
![Ảnh màn hình 2024-07-31 lúc 18 43 47](https://github.com/user-attachments/assets/01074e56-8cb8-4aee-bf95-543bbb770a66)

## 3. Ưu/khuyết điểm

**Ưu điểm**

- có thể tự viết công thức tính khoảng cách, miễn là thoả mãn 3 điều kiện sau

pt tính kcach khi tính khoảng cách giữa 2 điểm trùng nhau (A và chính nó) thì kết quả trả và phải bằng 0

nguyên tắc đối xứng: dA,B = dB,

positive: dA>=0

**Khuyết điểm**

- cruise of high dimensionality

khi khảo sát 1 tập dữ liệu có quá nhiều biến thì Độ chính xác của KNN có thể bị suy giảm nghiêm trọng với dữ liệu kích thước cao do có rất ít sự khác
biệt giữa hàng xóm gần nhất và xa nhất. -> có thể làm cho mô hình KNN hoạt động không tốt 

- ảnh hưởng bởi data imbalance

## 4. Xây dựng KNN sử dụng sklearn

- cải thiện bằng cách tăng chất lượng dataset (generate/collect, crawl dữ liệu, synthetic,...)

synthetic là sinh ra mẫu ão, hay còn gọi là augmentic đối với mẫu ảnh (ví dụ có thể flip, rotate lại độ để add vô dataset ban đầu để làm cho data set này tăng lên 

https://www.kaggle.com/datasets/crawford/emnist?select=emnist-mnist-test.csv

<img width="451" alt="Ảnh màn hình 2024-07-31 lúc 20 35 35" src="https://github.com/user-attachments/assets/031459b7-fb2b-4737-9c36-fe35213f5324">  <img width="455" alt="Ảnh màn hình 2024-07-31 lúc 20 35 50" src="https://github.com/user-attachments/assets/43c4bef4-7d53-494e-bde2-e18002ca4c0c">

# Buổi học 4: Supervised Learning - Linear Regression và Gradient Descent (2/08/2024)

# Gradient Descent

## 1. Caculus

- đạo hàm là rate of change
  
<img width="1051" alt="Ảnh màn hình 2024-08-02 lúc 18 38 29" src="https://github.com/user-attachments/assets/5a965220-b860-482f-84f7-cde81bcd6809">

## 2. Gradient Descent trong Python

- Thuật toán này dùng để tối ưu hoá 1 hàm số nào đó bằng cách sử dụng đạo hàm

- Gradient descent là xương sống của các thuật toán Machine Learning.

<img width="390" alt="Ảnh màn hình 2024-08-02 lúc 18 43 31" src="https://github.com/user-attachments/assets/f928a24c-64b9-43b9-bf68-eea70167ca21">

![Ảnh màn hình 2024-08-02 lúc 18 45 59](https://github.com/user-attachments/assets/8cf25517-a0d0-4887-bbf1-18a6a4b718e9)

# Linear Regression

## 1. Regression Analysis
## 2. Đánh giá mô hình Regression

<img width="879" alt="Ảnh màn hình 2024-08-02 lúc 19 08 41" src="https://github.com/user-attachments/assets/603e5bf2-0493-4c9f-a51f-3dfeee0aff98">

đối với những bài toán có range khác nhau 

<img width="685" alt="Ảnh màn hình 2024-08-02 lúc 19 10 58" src="https://github.com/user-attachments/assets/5ceaa152-dc4c-41cf-9dbc-c6253e0f984b">

<img width="877" alt="Ảnh màn hình 2024-08-02 lúc 19 13 08" src="https://github.com/user-attachments/assets/b0f204c7-4a86-4da4-b5ba-53b685ee143d">

## 3. Linear regression

- có thể dùng close form (công thức có sẵn) hoặc gradient decent để tính

## 4. Polynomial Regression

<img width="893" alt="Ảnh màn hình 2024-08-02 lúc 19 22 44" src="https://github.com/user-attachments/assets/5d21c9b9-c29e-445e-80e9-ea78f628d5bf">

 - giúp giải quyết vde của liner regression thông thường, thể hiện đc mối quan hệ phi tuyến tính

## 5. Multicol linearity

<img width="424" alt="Ảnh màn hình 2024-08-02 lúc 19 31 48" src="https://github.com/user-attachments/assets/391fcbfe-12f1-43b4-a6d4-ce883b8b0add">  <img width="421" alt="Ảnh màn hình 2024-08-02 lúc 19 32 00" src="https://github.com/user-attachments/assets/a5861aff-ba2f-4a72-a6e9-41efa8c1b78e">


