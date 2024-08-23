<img width="1038" alt="Ảnh màn hình 2024-08-09 lúc 15 58 20" src="https://github.com/user-attachments/assets/abf0fb6a-1dcb-4f20-b268-2de93a7af4c5"># Course structure

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

<img width="644" alt="Ảnh màn hình 2024-07-26 lúc 17 24 14" src="https://github.com/user-attachments/assets/2cc25be9-5aed-4404-b01e-689d1e4a2a2e">

<img width="637" alt="Ảnh màn hình 2024-07-26 lúc 17 24 19" src="https://github.com/user-attachments/assets/15aae597-ec5e-4783-8752-6bd4be461478">

<img width="786" alt="Ảnh màn hình 2024-07-26 lúc 17 24 24" src="https://github.com/user-attachments/assets/75117950-5a29-492a-a4cc-471c908318b0">

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

<img width="788" alt="Ảnh màn hình 2024-07-26 lúc 17 35 20" src="https://github.com/user-attachments/assets/e6be5c25-1225-4116-b5a6-6b394613ad15">

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

<img width="809" alt="Ảnh màn hình 2024-07-26 lúc 19 31 06" src="https://github.com/user-attachments/assets/8c7f1640-61f6-4777-bab7-9717c30f499d">

## 3. Thuật toán

### 1 số ứng dụng phổ biến có thể kể đến như 

- text classification (vd: phân loại bình luận, lọc thư rác,...)

- Bayesian Classification

- handwritten digit recognition

![Ảnh màn hình 2024-07-26 lúc 18 38 31](https://github.com/user-attachments/assets/eebcb6a0-490d-4627-a1a6-f9cded673e45)

### The Naïve Bayes Assumption

<img width="883" alt="Ảnh màn hình 2024-07-26 lúc 19 38 17" src="https://github.com/user-attachments/assets/b46c9bd9-d546-42df-a7c5-6aee659d1c13">

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


<img width="436" alt="Ảnh màn hình 2024-07-28 lúc 02 14 41" src="https://github.com/user-attachments/assets/912698f8-692f-495f-8eb1-358518cc86f4">

<img width="441" alt="Ảnh màn hình 2024-07-28 lúc 02 14 47" src="https://github.com/user-attachments/assets/2a94d71f-ef47-4194-91d9-f5c5b425a17b">

# Buổi học 3: Supervised Learning - K-Nearest Neighbors (28/07/2024)

https://www.gradio.app/guides/quickstart?authuser=1

https://www.mdpi.com/2227-7080/9/3/52?authuser=1

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

- Thuật toán này dùng để tối ưu hoá 1 hàm số nào đó bằng cách sử dụng đạo hàm. Mục tiêu của Gradient Descent là tìm giá trị tối thiểu của một hàm mất mát (loss function) bằng cách điều chỉnh các tham số của mô hình (ví dụ như các trọng số trong mạng nơ-ron).

- Gradient descent là xương sống của các thuật toán Machine Learning.

![Ảnh màn hình 2024-08-06 lúc 22 28 12](https://github.com/user-attachments/assets/9aeb6e60-ed64-4afb-a25b-7da725099144)

bước 3: update trọng số a và b sử dụng công thức sau 

<img width="390" alt="Ảnh màn hình 2024-08-02 lúc 18 43 31" src="https://github.com/user-attachments/assets/f928a24c-64b9-43b9-bf68-eea70167ca21">

**Cost (loss) function**

 - đo mức độ sai khác mà mô hình dự đoán so với giá trị thật. 2 trọng số a và b cần tìm là tại 2 trọng số đó giá trị của hàm cost function |y'-y|^2 là nhỏ nhất. lúc đó sai số của mô hình đưa ra cũng là nhỏ nhất. bài toán trở thành bài toán tìm a và b sao cho hàm số đạt GTNN

![Ảnh màn hình 2024-08-06 lúc 22 40 29](https://github.com/user-attachments/assets/15e8576a-09cd-414a-88ab-5330ac575dae)

-  α là learning rate sẽ có ảnh hưởng đến tốc độ trượt xuống đáy

learning rate quá nhỏ sẽ tốn thời gian lâu để tìm ra điểm cực tiểu của mô hình 

learning rate quá lớn sẽ có khả năng đi lố -> bỏ qua điểm cực tiểu 

# Linear Regression

## 1. Regression Analysis
## 2. Đánh giá mô hình Regression

- Bài toán regression là bài toán đoán ra một output y nào đó có vô vàn giá trị thật (giá nhà, dự đoán nhu cầu của sản phẩm,..)

- không dùng accuracy scrore để đánh giá hiệu quả của mô hình regression.

**thay vào đó dùng các phép đo sau đây để đánh giá:**

<img width="779" alt="Ảnh màn hình 2024-08-02 lúc 19 08 41" src="https://github.com/user-attachments/assets/603e5bf2-0493-4c9f-a51f-3dfeee0aff98">

- tuy nhiên, 2 metrics này có 1 vấn đề đó là sẽ không chính xác đối với những bài toán có đơn vị khômg đồng nhất và range không khớp với nhau.

- MSE, MAE chỉ dùng để so sánh sai số tuyệt đối

**Những phép đo dựa trên giá trị phần trăm sẽ không bị ảnh hưởng**

- Mean absolute percentage error

![Ảnh màn hình 2024-08-06 lúc 23 35 03](https://github.com/user-attachments/assets/adf42b0b-fbc4-4652-89bc-edf0d74a33d5)

<img width="685" alt="Ảnh màn hình 2024-08-02 lúc 19 10 58" src="https://github.com/user-attachments/assets/5ceaa152-dc4c-41cf-9dbc-c6253e0f984b">

<img width="777" alt="Ảnh màn hình 2024-08-02 lúc 19 13 08" src="https://github.com/user-attachments/assets/b0f204c7-4a86-4da4-b5ba-53b685ee143d">

## 3. Linear regression

- Thuật toán cơ bản thuộc nhóm Supervised Learning (Học có giám sát).
 
- Được sử dụng rất rộng rãi trong Regression Analysis (Phân tích hồi quy).

- Mô hình được “xây dựng” bằng cách sử dụng phương pháp Bình phương tối thiểu (Least Squares method).

phuong pháp này có thể dùng close form (công thức có sẵn) hoặc gradient decent để tính

**khuyết điểm**

- rất nhạy cảm với nhiễu và cần phải giải quyết nhiễu trước khi làm liner regression

![Ảnh màn hình 2024-08-06 lúc 23 47 18](https://github.com/user-attachments/assets/a37f7ea4-a6d6-4316-84b5-8685c8d84f2d)

- đối với những mô hình phức tạp hơn thì không phù hợp để sử dụng mô hình liner regression 

## 4. Polynomial Regression

- khác liner ở chỗ poly có bậc. có thể mô hình được môi quan hệ phi tuyến tính, ví dụ như:

<img width="432" alt="Ảnh màn hình 2024-08-06 lúc 23 50 04" src="https://github.com/user-attachments/assets/c722789f-ff74-43f8-9889-ab945c400384">

<img width="693" alt="Ảnh màn hình 2024-08-02 lúc 19 22 44" src="https://github.com/user-attachments/assets/5d21c9b9-c29e-445e-80e9-ea78f628d5bf">

 - giúp giải quyết vde của liner regression thông thường, thể hiện đc mối quan hệ phi tuyến tính

## 5. Multicol linearity

<img width="524" alt="Ảnh màn hình 2024-08-02 lúc 19 31 48" src="https://github.com/user-attachments/assets/391fcbfe-12f1-43b4-a6d4-ce883b8b0add">  <img width="421" alt="Ảnh màn hình 2024-08-02 lúc 19 32 00" src="https://github.com/user-attachments/assets/a5861aff-ba2f-4a72-a6e9-41efa8c1b78e">

## DEMO

https://www.kaggle.com/code/deanmendes/linear-regression-usa-housing?authuser=0

- xong bài toán regression thì phaải xem được ảnh hưởng của từng feature đối với biến output

- nên áp dụng normalize với linear regression -> cơ sở công bằng để đánh giá ảnh hưởng của từng biến lên mô hình

- phải có hint hoặc assumption trc khi áp dụng liner. trong trường hợp này thường dùng residual plot (residual plot là kĩ thuật để nhận ra giũa các biến này với biến target có mỗi quan hệ tuyến tính hay không)

- khi gặp hiện tượng các điểm data points randomly spread xung quanh trục x thì chứng tỏ có một liner relation tồn tại giữa 2 biến này

<img width="521" alt="Ảnh màn hình 2024-08-07 lúc 07 21 58" src="https://github.com/user-attachments/assets/a2c86e0d-a845-4fb4-9f51-4dac9501dc21">

- giả sử có mỗi qhe tuyến tính giữa biến price và biến area population thì nó sẽ tìm 1 ptrinh liner regreesion đơn giản để giải thích mói quan hệ tuyến tính này. sau đó so sánh kết quả y' với biến y thực tế. giả sử data set có 1000 mẫu sẽ tính ra đc 1000 điểm chệnh lệch. từ 1000 điểm chênh lệch đó sẽ dùng để vẽ nên đồ thị residal plot này. khi đồ thị này có dạng random thì những điểm trong đồ thị này đến từ các điểm không mô hình hoá đc 

- train_test_split trong thư viện scikit-learn

test_size: Tỷ lệ hoặc số lượng mẫu sẽ được tách ra để làm tập kiểm tra. Ví dụ, nếu test_size=0.2, 20% dữ liệu sẽ được dùng làm tập kiểm tra, và 80% còn lại sẽ dùng để huấn luyện.

train_size: Tỷ lệ hoặc số lượng mẫu sẽ được tách ra để làm tập huấn luyện. Nếu không được chỉ định, phần còn lại sau khi tách test_size sẽ được dùng làm tập huấn luyện.

random_state: Số nguyên ngẫu nhiên được sử dụng để đảm bảo tính tái lập. Khi bạn chạy lại mã với cùng một 
random_state, bạn sẽ nhận được cùng một sự phân chia dữ liệu.

shuffle: Mặc định là True, dữ liệu sẽ được xáo trộn trước khi phân chia để đảm bảo tính ngẫu nhiên.

## techtalks

- Personality Indentification using Deep Learning

- Face Recognition
  
- AUC-ROC curve & ứng dụng

# Buổi học 5: Supervised Learning - Logistic Regression + Pipeline & Pandas Profilling (4/08/2024)

https://www.datacamp.com/tutorial/pandas-profiling-ydata-profiling-in-python-guide?authuser=0

https://www.kdnuggets.com/2020/03/linear-logistic-regression-explained.html?authuser=0

https://www.kaggle.com/code/ericle3121/logistic-regression-for-heart-disease-classificati

https://www.kaggle.com/code/alexisbcook/pipelines?authuser=0

- tên là regression nhưng chủ yếu để làm classification task

## 1. Classification

- biến output là biến categorical (hữu hạn gía trị có thể có)

- classification là bài toán supervised (cần data có nhãn)

**Phân loại**

<img width="631" alt="Ảnh màn hình 2024-08-04 lúc 08 41 13" src="https://github.com/user-attachments/assets/b21d5f33-20db-4421-bfc7-9e228a21cb4b">

## 2. Đánh giá mô hình Classification

- cần có 2 thứ để đánh giá: giá trị thực ngoài đời và giá trị dự đoán

giá trị thực (actual values–y) và giá trị dự báo (predicted values – y predict).

**Các thang đo mô hình classification**

<img width="585" alt="Ảnh màn hình 2024-08-04 lúc 08 43 09" src="https://github.com/user-attachments/assets/daf2cbb0-0d9c-46a8-abdf-d6b8468e01de">

- Khi dữ liệu bị mất cân bằng (imbalanced data), accuracy để đánh giá sẽ không đáng tin 

- Precision và Recall sử dụng trong trường hợp data bị imbalance

Recall dùng khi tiêu chí đánh giá là "nhầm còn hơn sót" -> ví dụ dự đoán ngừoi đó có bị tiểu đường hay không. Trong 100 ngừoi bị tiểu đường thì mô hình dự đoán đúng bao nhiêu người 

Precision thể hiện "độ tự tin" của mô hình. ví dụ trong 100 ngừoi bị tiểu thì 90% trong số đó là thực sự bị tiểu đường 

-  F1score là metrics cân bàng giữa Precision và Recall . trường hợp muốn cân bằng cả 2 thì sử dụng F1score

**ROC-AUC**

- AUC càng cao thì càng tốt. AUC Càng cao thì mô hình càng robust

- thầy có gưỉ tài liệu tham khảo AUC.

**Visualize class data**

![Ảnh màn hình 2024-08-04 lúc 08 57 07](https://github.com/user-attachments/assets/9416b81b-ee70-4ab6-bf8e-9b63e0123ebd)

## 3. Logistic Regression

<img width="833" alt="Ảnh màn hình 2024-08-04 lúc 08 57 48" src="https://github.com/user-attachments/assets/03567e0f-71c3-475d-b34b-827452195342">

- Hàm sigmoid là hàm làm cho linear regression trở thành mô hình logistic regression (dự đoán ra 2 giá trị nằm trong khoảng 0 -> 1)

![Ảnh màn hình 2024-08-04 lúc 09 03 42](https://github.com/user-attachments/assets/6ae68030-c55b-4836-afb6-73fc0f434987)

**Ưu điểm**

- Dễ dàng mở rộng cho bài toán target có nhiều hơn hai loại

- Huấn luyện nhanh, độ chính xác cao cho nhiều tập dữ liệu đơn giản

- Có thể giải thích các hệ số mô hình cũng như các chỉ số về tầm quan trọng của tính năng

**Khuyết điểm**
- Decision boundary: chỉ phân biệt  với trường hợp 2 nhãn là  linearly separable. đôi với đường non-linear thì hoạt động không tốt.

<img width="604" alt="Ảnh màn hình 2024-08-04 lúc 09 05 29" src="https://github.com/user-attachments/assets/515f31ba-8ac9-4bda-907f-cbf592f31145">

- cách để biết nhãn của mình có bị non linear không? 

C1: cách thử: cứ áp dụng log regression, nếu làm k tốt -> non linear

C2: dùng kỹ thuật pca để giảm không gian về 2 chiều và visualize

## 4. Hiệu chỉnh ngưỡng (threshold) trong Classification

## 5. Pipelines

https://www.kaggle.com/code/alexisbcook/pipelines?authuser=0

- Pipelines are a simple way to keep your data preprocessing and modeling code organized.

ghép những step trong preprocessing, step trong mô hình vào chung 1 chỗ để cho code gọn hơn 

<img width="489" alt="Ảnh màn hình 2024-08-04 lúc 10 05 58" src="https://github.com/user-attachments/assets/11db227c-4567-4bee-91e1-a39398dc13f9"> <img width="493" alt="Ảnh màn hình 2024-08-04 lúc 10 04 37" src="https://github.com/user-attachments/assets/982ade70-2a5f-4fa0-9b18-4b6dea8ad267">

# Buổi học 6: Supervised Learning - Decision Tree & Random Forest (7/08/2024)

https://www.datascienceprophet.com/understanding-the-mathematics-behind-the-decision-tree-algorithm-part-i/?authuser=1

https://www.kaggle.com/code/sravan1701/churn-prediction-using-random-forest-and-smote?authuser=1

https://www.kaggle.com/code/hongngcthuthng/decision-tree-random-forest-with-classweight#3.-Test-the-3-models

https://varshasaini.in/glossary/smote/?authuser=1

# Decision Tree

## 1.Giới thiệu

- vẽ ra 1 sơ đồ đi từ thời điểm ban đầu đến đích dựa vào việc trả lời các câu hỏi đưa ra

- được sử dụng khá nhiều trong lĩnh vực finance

- có tính giải thích cao (interpretibility)

- decision tree là base model (thuật toán nền) để build mô hình random forest 

<img width="430" alt="Ảnh màn hình 2024-08-09 lúc 07 25 54" src="https://github.com/user-attachments/assets/03714ac8-1d0a-4914-8787-72fcf973c4b3">

- đặc biệt đc dùng nhiều trong phân tích quyết định

- <img width="581" alt="Ảnh màn hình 2024-08-09 lúc 08 09 18" src="https://github.com/user-attachments/assets/77888e1e-c744-4afe-b89b-fb14efd186c0">

- ở trong noole sẽ khảo sát các giả trị có thể có của 1 biến, decison tree sẽ tìm cách chọn ra biến nào đặt vô node noà cho phù hợp

noole trên cùng đầu tiên gọi là root noole

tiêu chí lựa chọn biến để đưa vào noole của decison tree dựa trên "thuần khiết"

**Làm thế nào để xác định được phân chia tốt nhất?

<img width="423" alt="Ảnh màn hình 2024-08-09 lúc 08 18 09" src="https://github.com/user-attachments/assets/29aa7e59-6f0f-400b-98c2-b58b627fa19b">

- sự thuần khiết của 1 biến đc định nghĩa bằng cách chia sau khi chia data set theo biến đó.

## 2. Thuật toán

**Đo mức độ không thuần khiết**

- để máy tính hiểu biến nào "thuần khiết" hơn thì cần sự dụng 1 phép đo để đo mức độ không đồng nhất khi cắt dữ liệu trong 1 node và thuật toán đó được gọi là gini index

<img width="469" alt="Ảnh màn hình 2024-08-09 lúc 08 21 34" src="https://github.com/user-attachments/assets/e200c679-1dd5-4814-81ca-90362cfd59dd">\

<img width="270" alt="Ảnh màn hình 2024-08-09 lúc 08 22 51" src="https://github.com/user-attachments/assets/2c6c8a5e-382f-413d-8f35-a3bb507a12fa">

- gini thuộc R, gini thấp -> tính đồng nhất cao và ngược lại

- một số người không thích dùng gini thì có thể dùng Information Gain. khác nhau ở chỗ sẽ lựa chọn biến có information gain cao nhất. 

## 3 Ưu/khuyết điểm

**Ưu điểm**

<img width="421" alt="Ảnh màn hình 2024-08-09 lúc 09 34 50" src="https://github.com/user-attachments/assets/86d7ea51-49eb-4122-a82a-35c9bd3c5576">

- Khi áp dụng trong regression (kết quả ra là một con số chứ không phải là phân loại) trong decision tree thì cả gini và information gain đều k dùng đc. Vì vậy cần áp dụng mô hình có cách hdong tương tự có tên là 'decision tree regression', mô hình này sử dụng các metrics như absolute_error, MSE,.. để đo sự giống nhau giữa những con số thực 

<img width="649" alt="Ảnh màn hình 2024-08-09 lúc 09 40 33" src="https://github.com/user-attachments/assets/61f30ce6-6083-42f8-bce0-c249b0265c39">

**Khuyết điểm**

<img width="420" alt="Ảnh màn hình 2024-08-09 lúc 09 48 44" src="https://github.com/user-attachments/assets/e4dd9944-f77c-4591-ba28-0791333afce4">

- nếu có biến continous thì không nên dùng decision tree, hoặc drop thử biến rồi xem decision tree còn hiệu quả k r mới dùng

## 4. Xây dựng Decision Trees sử dụng sklearn

#  Ranđom Forest

## 1.Giới thiệu

- random forest là một trong những mô hình manh nhất của machine learning

- Nhược điểm chính của cây quyết định (Decision Tree) là chúng có khuynh hướng overfit dữ liệu huấn luyện.

- Random Forest là một cách để giải quyết vấn đề này. Một Random Forest về bản chất là một tập hợp các cây quyết định, trong đó mỗi cây hơi khác so với các cây khác.

**Ứng dụng**

<img width="1040" alt="Ảnh màn hình 2024-08-09 lúc 15 58 39" src="https://github.com/user-attachments/assets/64e22cc4-e063-4ae2-96d5-0586e6530193">

## 2. Thuật toán

- Khi dự đoán, Random Forest sẽ kết hợp các dự đoán từ từng cây bằng cách sử dụng phương pháp bình chọn (major voting) cho bài toán classification hoặc trung bình (averaging) cho bài toán regression.

## 3 Ưu/khuyếtđiểm

**Ưu điểm**

- ta có thể Nhận thấy tầm quan trọng của tính năng tương đối, giúp chọn các tính năng đóng góp nhiều nhất cho quá trình phân loại.

search "random forest feature importance."

- ngoài ra chúng ta còn có thể tránh nhiều bằng cách remove bớt những thuộc tính có % đóng góp vào mô hình quá thấp. điều này cũng sẽ góp phần làm mô hình không bị nhiễu

- biểu đồ đánh giá mức độ đóng góp cuả từng feature đối với hô hình -> hỗ trợ chúng ta lựa chọn feature selection dựa trên feature importantce. randome forest làm cách này bằng cách bỏ các feature ra khỏi mô hình. nếu feature đó làm mô hình sụt gỉam điểm số nhiều thì đó là feature quan trọng

![Ảnh màn hình 2024-08-09 lúc 16 13 50](https://github.com/user-attachments/assets/d984f6dd-e46e-4c85-bd76-f934103f4d87)

**Khuyết điểm**

- chậm, tính giải thích thấp

## 4. Xây dựng Random Forests sử dụng sklearn

### DEMO 1: Decision Tree & Random Forest with ClassWeight

- ClassWeight là một kỹ thuật dùng để chống data imbalance bằng cách gán cho nhãn minority số điểm cao hơn. điều này giúp mô hình quan tâm nhiều hơn và học nhiều hơn đối với nhãn ít mẫu (ví dụ 90% nhãn survive 10% nhãn die). Class Weight: Phương pháp này liên quan đến việc điều chỉnh trọng số của các lớp trong quá trình huấn luyện mô hình. Điều này hữu ích khi bạn có dữ liệu không cân bằng, và bạn muốn mô hình tập trung hơn vào các lớp ít phổ biến hơn bằng cách tăng trọng số của chúng trong hàm mất mát.

- nếu kaggle không cho chạy pandas profiling thì có thể dùng ydata-profiling cũng tương tự

- ngoài mất cân bằng giữa các lable còn có trường hợp mất cân bằng nặng hơn là mất cân bằng cả trong pattern của lable đó. Các cách xử lý: xử lý outliers, over sampling,... -> thầy khuyên dùng cách over sampling = phương pháp SMOTE để an toàn hơn cho mô hình

btvn: edit notebook này dựa trên gợi ý feature importance và oversampling (có thể sử dụng pp SMOTE) để cải thiện model này

- Feature Importance: Đây là quá trình xác định và đánh giá tầm quan trọng của các đặc trưng (features) trong mô hình. Từ đó, bạn có thể chọn lọc những đặc trưng quan trọng nhất hoặc giảm bớt những đặc trưng ít quan trọng để giảm thiểu sự phức tạp của mô hình và cải thiện hiệu suất.

### DEMO 2: Churn prediction using Random Forest and SMOTE

https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
 
- SMOTE stands for Synthetic Minority Oversampling Technique. kỹ thuật ov sampling dựa trên KNN, tập trung vào nhãn minority.

- SMOTE (Synthetic Minority Over-sampling Technique): Đây là một kỹ thuật dùng để xử lý vấn đề dữ liệu không cân bằng bằng cách tạo ra các mẫu mới từ lớp thiểu số. SMOTE tạo ra các mẫu tổng hợp bằng cách kết hợp các điểm gần nhau trong không gian đặc trưng, giúp cân bằng lại tỉ lệ giữa các lớp trong dữ liệu huấn luyện.
 
- nói về cách sử dụng pp SMOTE trên mô hình

<img width="455" alt="Ảnh màn hình 2024-08-08 lúc 10 34 00" src="https://github.com/user-attachments/assets/b5a3fbec-37aa-4878-814a-39b1dfbee864"> <img width="452" alt="Ảnh màn hình 2024-08-08 lúc 10 34 10" src="https://github.com/user-attachments/assets/ce13f325-64e3-4f80-9176-6b23410e72e8">

# Buổi học 7: Supervised Learning - SVM (9/08/2024)

## 1. Giới thiệu

- SVM (Support Vector Machine) là một mô hình Machine Learning rất mạnh mẽ và linh hoạt, có khả năng thực hiện phân loại tuyến tính hoặc phi
tuyển, hồi quy và thậm chí phát hiện ngoại lệ.

- SVM là một trong những mô hình phổ biến nhất trong Machine Learning, và bất cứ ai quan tâm đến Machine Learning nên biết.

- SVM đặc biệt thích hợp để phân loại các bộ dữ liệu phức tạp nhưng nhỏ hoặc vừa. (vì SVM chạy khá lâu)

- Ngoài việc thực hiện phân loại tuyển tính, SVM có thể thực hiện hiệu quả phân loại phi tuyến tính bằng cách sử dụng "Kernel trick", ngầm ánh xạ các input vào không gian high-dimensional feature. (tránh được cruise of high dimensionality)

## 2. Thuật toán

- Trong machine learning, nhiệm vụ của mô hình SVM (Support Vector Machine) là phân loại dữ liệu. Cụ thể, SVM tìm ra một siêu phẳng (hyperplane) tốt nhất để phân chia dữ liệu thành các lớp khác nhau. Siêu phẳng này được chọn sao cho khoảng cách (margin) giữa nó và các điểm dữ liệu gần nhất của mỗi lớp là lớn nhất.

<img width="636" alt="Ảnh màn hình 2024-08-19 lúc 14 45 26" src="https://github.com/user-attachments/assets/22a8142e-1efc-48f9-aa28-20f8da338148">

- SVM phát biểu : bất kì mô hình nào có margin space rộng nhất thì sẽ là mô hình tốt nhất. Lý do: khi margin space lớn thì khả năng giảm nhiễu cao, giảm khả năng phân loại sai 

**Hard-margin SVM (Linear SVM)**

<img width="420" alt="Ảnh màn hình 2024-08-19 lúc 14 59 08" src="https://github.com/user-attachments/assets/74c19ad1-51c5-4ca1-8e4b-f0bc7cc8e5b3">

Hard-margin SVM là một phiên bản của SVM được sử dụng khi dữ liệu có thể được phân tách hoàn toàn bằng một đường thẳng (hoặc siêu phẳng) mà không có lỗi. Cách nó hoạt động như sau:

- Phân tách tuyến tính: Hard-margin SVM tìm kiếm một siêu phẳng (hyperplane) phân tách hoàn toàn hai lớp dữ liệu mà không có điểm nào bị phân loại sai.

- Maximizing the Margin: SVM không chỉ tìm bất kỳ siêu phẳng nào mà cố gắng tối đa hóa khoảng cách (margin) giữa siêu phẳng đó và các điểm gần nhất thuộc hai lớp (các support vectors).

- Không chấp nhận lỗi: Trong Hard-margin SVM, không có dữ liệu nào được phép nằm giữa hoặc bên phía sai của siêu phẳng. Điều này nghĩa là mô hình chỉ hoạt động tốt khi dữ liệu hoàn toàn có thể phân tách.

Vì vậy, Hard-margin SVM thích hợp cho các tập dữ liệu mà hai lớp có thể được phân tách hoàn toàn mà không có bất kỳ sự chồng chéo nào.

**Soft margin SVM**

<img width="581" alt="Ảnh màn hình 2024-08-19 lúc 14 59 46" src="https://github.com/user-attachments/assets/853510dc-6e48-4a4e-bad6-4bf4be39f5bc">

Soft-margin SVM là một phiên bản của SVM cho phép một số lỗi phân loại để tạo ra một mô hình linh hoạt hơn. Cách nó hoạt động như sau:

- Cho phép lỗi: Soft-margin SVM cho phép một số điểm dữ liệu nằm sai phía của siêu phẳng (hyperplane) hoặc nằm trong vùng margin giữa hai lớp. Điều này giúp mô hình hoạt động tốt hơn khi dữ liệu không thể phân tách hoàn toàn.

- Cân bằng giữa lỗi và margin: Mô hình cố gắng tối đa hóa khoảng cách (margin) giữa siêu phẳng và các điểm dữ liệu gần nhất, nhưng đồng thời cũng cho phép một số điểm vi phạm margin (tức là bị phân loại sai). Số lượng và mức độ vi phạm được điều chỉnh bởi một tham số C, giúp cân bằng giữa việc có một margin lớn và số lượng lỗi.

- Tính linh hoạt: Nhờ việc cho phép một số điểm dữ liệu bị phân loại sai, Soft-margin SVM linh hoạt hơn trong việc xử lý các tập dữ liệu có nhiễu hoặc không thể phân tách hoàn toàn.

Tóm lại, Soft-margin SVM giúp mô hình không quá cứng nhắc, làm việc tốt hơn với dữ liệu thực tế có chứa nhiễu hoặc không thể phân tách rõ ràng.

**The “Kernel Trick”**

- một số tình huống ở không gian thấp chiều thì các điểm không phân tách nhau ra rõ ràng đươch thế những khi đưa các điểm đó vào không gian 3 chiều thì lại có thể phân tách rõ ràng được

<img width="716" alt="Ảnh màn hình 2024-08-19 lúc 15 01 49" src="https://github.com/user-attachments/assets/2d553dcd-01c2-42d2-8a9c-7fb111bcc11f">

## 3. Ưu/khuyết điểm

**Ưu điểm**

<img width="843" alt="Ảnh màn hình 2024-08-09 lúc 19 00 42" src="https://github.com/user-attachments/assets/22acbb49-c53b-4f43-94f8-6371d344fbf0">

- thường là mô hình đầu tay để dùng khảo sát độ khó trên tập dữ liệu

- nếu áp dụng svm cho ra điểm số thấp thì chúng ta có thể thấy được đây là một bài toán thật sự phức tạp

**Khuyết điểm**

<img width="700" alt="Ảnh màn hình 2024-08-19 lúc 15 05 17" src="https://github.com/user-attachments/assets/aaa472d6-6deb-42e6-843c-6f32cbb5d790">

## 4. Xây dựng SVMs sử dụng sklearn

# Buổi học 7: Boosting Techniques (9/08/2024)

### Giới thiệu Boosting

- Boosting" hỗ trợ cho các mô hình Machine Learning để cải thiện độ chính xác của dự đoán.

- Boosting algorithm (Thuật toán tăng cường) là một trong những thuật toán được sử dụng khá rộng rãi nhằm mục đích tăng cường thuật toán để cải thiện độ chính xác của các mô hình.
 
- Đây là một thuật toán học quần thể bằng cách xây dựng nhiều thuật toán cùng lúc (ví dụ như Decision Tree) và kết hợp chúng lại. Mục đích là để có một cụm hoặc một nhóm các 'weak learner' rồi kết hợp chúng lại để tạo ra một 'strong learner' duy nhất.

## 1. AdaBoost (Adaptive Boosting)

- AdaBoost (Adaptive Boosting) là một kỹ thuật boosting phổ biến giúp ta kết hợp nhiều 'weak classifier' (trình phân loại yếu) thành một 'strong classifier' (trình phân loại mạnh) duy nhất.

- base model có thể là bất kì mô hình machine learning nào

- hoạt động trên tư duy sửa lỗi: có 1 chuỗi model nối tiếp nhau, model sau sẽ học tập dựa trên sai lầm của model trước.

### Thuật toán AdaBoost

![Ảnh màn hình 2024-08-19 lúc 15 22 12](https://github.com/user-attachments/assets/f77d3c06-1927-4e2a-9679-c250fcdf1350)

![Ảnh màn hình 2024-08-19 lúc 15 23 19](https://github.com/user-attachments/assets/d44807a5-01f8-4e35-b339-4e8a2b38e8fb)

<img width="525" alt="Ảnh màn hình 2024-08-19 lúc 15 25 52" src="https://github.com/user-attachments/assets/4de54674-09a0-42c5-a476-0b3f21e802ac">

## 2. XGBoost (thuộc nhóm Gradient Boosting(

https://www.youtube.com/watch?v=PxgVFp5a0E4&t=14s

- XGBoost không chỉ là một model mà còn là 1 thư viện machine learning like numpy, tensorflow, pytorch.

- XGBoost là One of the Popular Tools of Winners is XGBoost.
 
- base model của XGBoost là decision tree

- hoạt động trên tư duy sửa lỗi: model sau sẽ DỰ ĐOÁN sai lầm của model trước

- XGBoost là một thuật toán "boosting," có nghĩa là nó xây dựng một loạt các cây quyết định (decision trees) theo tuần tự, mỗi cây mới được thêm vào để sửa lỗi của cây trước đó.

![Ảnh màn hình 2024-08-19 lúc 16 22 54](https://github.com/user-attachments/assets/98cd1a17-05ac-4646-bce8-01c805074763)

## 3. LightGBM

https://sefiks.com/2020/05/13/xgboost-vs-lightgbm/

- XGBoost khi train từng cây decision tree thì sử dụng tư duy level wise còn Light BGM sử dụng tư duy  leaf wise

**Level-wise (XGBoost)**

- Khi train 1 cây decision tree mà muốn rẽ xuống nhánh phía dứoi thì phải đợi cho tất cả các nhánh (node) phía trên hoàn thành hết thì mới được di chuyển xuống phía dưới

**Leaf-wise (Light GBM)**

- Leaf-wise nghĩa là LightGBM mở rộng cây bằng cách chọn nút lá có độ lợi cao nhất (largest gain) để chia nhỏ trước, thay vì mở rộng tất cả các nút cùng cấp độ.

- Light GBM dùng tư duy leaf-wise nên sẽ nhanh hơn XGBoost khoảng chừng 10 lần

Trong điều kiện lý tưởng với thời gian và tài nguyên vô hạn, cả Level-wise (XGBoost) và Leaf-wise (LightGBM) có thể đạt được kết quả tương tự, vì cả hai sẽ xây dựng được các cây quyết định tốt nhất có thể để tối ưu hóa hiệu suất mô hình. Tuy nhiên, nếu bạn giới hạn số lượng iterations (ví dụ: chỉ cho phép 300 iterations), kết quả sẽ khác nhau 

**Khuyết điểm của Light GBM**

- dễ bị overfitting hơn XGBoost

## 4. CatBoost 

https://www.youtube.com/watch?v=KXOTSkPL2X4

![Ảnh màn hình 2024-08-19 lúc 20 33 41](https://github.com/user-attachments/assets/dfc15462-3d7f-42aa-9b65-cbd8298cffe9)

![Ảnh màn hình 2024-08-19 lúc 20 36 49](https://github.com/user-attachments/assets/363e9107-e869-49b5-9a89-773509129797)

- CatBoost Encoder sử dụng Ordered Target Mean Encoding. Điều này có nghĩa là nó chỉ sử dụng các thông tin từ dữ liệu trước đó để mã hóa từng điểm dữ liệu hiện tại, giúp tránh việc mô hình học thuộc (overfitting) từ toàn bộ dữ liệu huấn luyện.
 
- CatBoost tiên tiến hơn trong xử lý biến categorical nhờ khả năng xử lý trực tiếp mà không cần mã hóa, sử dụng phương pháp Ordered Boosting để giảm overfitting, và tích hợp các kỹ thuật giảm sai số, giúp nó hoạt động hiệu quả hơn trên các tập dữ liệu có nhiều biến phân loại.

- cơ chế xử lý biến categorical

**Ích lợi**

- Tăng cường sự khác biệt giữa các mẫu -> tăng cường mối quan hệ giữa target variable -> tăng cường độ chính xác cho mô hình

- Khi chuyển về giá trị số sẽ dễ dàng áp dụng các phương pháp áp dụng cho biến numberical (mean, mode,...)

### DEMO 1: SVM vs XGBoost vs Random Forest

https://scikit-learn.org/stable/api/sklearn.svm.html

https://www.kaggle.com/code/hongngcthuthng/svm-vs-xgboost-vs-random-forest

<img width="565" alt="Ảnh màn hình 2024-08-19 lúc 20 59 42" src="https://github.com/user-attachments/assets/369ce681-1fcc-4774-bfb3-89dc7ba0f2d3">

- khi rơi vào 2 tình huống:

tình huống A: all feature -> 95 điểm 

tình huống B: drop 4 feature -> 95 điểm 

-> vẫn nên drop vì 4 feature đó không quá quan trọng vì vẫn giữ nguyên performance, drop sẽ giúp mô hình chạy nhanh hơn 

### DEMO 2: Comparison of catboost and one-hot encoding

https://www.kaggle.com/code/hongngcthuthng/comparison-of-catboost-and-one-hot-encoding

**Cách dùng Cat boost**
https://www.geeksforgeeks.org/categorical-encoding-with-catboost-encoder/

- This may prove that when we use random forest, GBDT, XGBoost, LightGBM and other tree model classification, we should probably reduce the use of one-hot encoding.

=> từ sau khi sử dụng các mô hình như random forest, GBDT, XGBoost, LightGBM thì đừng dùng các cách mã hoá như bình thường nữa mà hãy sử dụng catboost catboost encoding

<img width="497" alt="Ảnh màn hình 2024-08-09 lúc 20 50 04" src="https://github.com/user-attachments/assets/10911f8b-acf0-4655-af68-d2067985fcd8">

<img width="491" alt="Ảnh màn hình 2024-08-09 lúc 20 50 12" src="https://github.com/user-attachments/assets/8974568f-2700-4fd2-b1b6-63771381991c">

# Buổi học 8: Neural Network (10/08/2024)

https://paperswithcode.com

https://pytorch.org

- những nghiên cứ mới nhất về AI & ML

https://www.deeplearningbook.org/contents/numerical.html

## 1. Giới thiệu về Pytorch

**Một số tính năng nổi trội của Pytorch**

- Native ONNX Support

- Flexible Data Classes & DataLoader

**Tensor**

- Thành phần cơ bản nhất của Pytorch là Tensor.

- Về cơ bản, Tensor tương tự như Ndarray trong Numpy, nhưng có thể được tính toán nhanh hơn nhờ vào việc tăng tốc với GPU.

**Tensor & Vector**

![Ảnh màn hình 2024-08-22 lúc 14 02 58](https://github.com/user-attachments/assets/3f493a99-4397-4b5a-8f9f-c97d972b0a5a)

**Tensor & Differentiation(Tính đạo hàm với Pytorch)**

<img width="1267" alt="Ảnh màn hình 2024-08-10 lúc 09 03 15" src="https://github.com/user-attachments/assets/f325b4db-2853-43da-aa74-f2ad2d4b84ba">

f.backward() là câu lệnh tự tính đạo hàm

## 2. Giới thiệu về Neural Network

- Neural Network là 1 thuật toán được đề xuất để mô phỏng hoạt động của các neural thần kinh trong bộ não con người.

- Đây là thuật toán lõi trong deep learning

- Mạng neural network đơn giản nhất được gọi là Perceptron.

- Cấu tạo của 1 mạng neural network gồm 1 lớp input, 1 lớp output và 1 hoặc nhiều lớp hidden.

- Mỗi node trong 1 lớp còn được gọi là 1 neural, mỗi neural trong lớp hiện tại sẽ được link tới toàn bộ neural của lớp tiếp theo, tạo thành 1 mạng lưới neural thần kinh dày đặc (dense neural network).

**Cấu tạo của 1 neural network**

![Ảnh màn hình 2024-08-22 lúc 14 15 57](https://github.com/user-attachments/assets/dab3258b-5b5a-49cd-9ebb-4af70ff8974f)

- Chúng ta tiến hành mổ xẻ cách hoạt động của 1 neural:

1 neural gồm bộ phận tiếp nhận và bộ phận trả kết quả. bộ phận trả kết qủa sẽ có nhiệm vụ lấy kết quả mà bộ phận tiếp nhận vừa tính đem đi qua hàm g (hàm sidmoid) để kích hoạt kết quả đó. 

<img width="886" alt="Ảnh màn hình 2024-08-22 lúc 14 21 55" src="https://github.com/user-attachments/assets/925d9fba-e44f-4460-9b96-9e28f2d611d2">

![Ảnh màn hình 2024-08-22 lúc 14 23 17](https://github.com/user-attachments/assets/790302d9-1e9b-48f3-8f17-a5b03c02ac52)

**Tại sao chúng ta cần có Activation Function (hàm g)**

- kết quả của hàm g chính là output của neural
 
- Vai trò của Activation Function: đưa sự phi tuyến vào mô hình, chuẩn hoá giá trị đầu ra, tăng tốc độ học,..

**Forward Propagation**

- Forward Propagation là quá trình tính toán đầu ra của mạng dựa trên đầu vào. Quá trình này diễn ra từ lớp đầu vào (input layer), qua các lớp ẩn (hidden layers), và cuối cùng đến lớp đầu ra (output layer). Mỗi nút trong lớp này sẽ nhận giá trị từ các nút của lớp trước đó, nhân với các trọng số (weights), cộng thêm một bias, sau đó áp dụng một hàm kích hoạt (activation function) để tính toán giá trị đầu ra.

- Cơ chế feed forward giúp tính toán giá trị output của mạng neural network.

## 3. Thuật toán Back Propagation

- còn gọi là loss function, objective function, cost function

- Tác dụng: nhận nhãn thực tế và kết quả mô hình dự đoán (y,y^) = numerical = z. khi độ lỗi/sai số lớn thì z càng lớn và ngược lại. thuật toán này sẽ tìm bộ trọng số thích hợp sao cho giá trị "Loss Value" (hay còn gọi là hàm lỗi) càng nhỏ càng tốt

- Cơ chế backpropagation giúp tính toán các đạo hàm cần thiết, để từ đó Gradient Descent sẽ update các trọng số trong mô hình.

<img width="685" alt="Ảnh màn hình 2024-08-10 lúc 09 24 31" src="https://github.com/user-attachments/assets/01e649fb-f862-46e4-8574-1ac91c7f0464">

**Stochastic Gradient Descent**

<img width="1216" alt="Ảnh màn hình 2024-08-10 lúc 10 07 57" src="https://github.com/user-attachments/assets/3b04384d-4dd0-44fa-9a5e-81c71b86587b">

**Batch Gradient Descent**

<img width="1032" alt="Ảnh màn hình 2024-08-22 lúc 15 12 10" src="https://github.com/user-attachments/assets/3921a68d-390c-450b-a230-52b6c45178e3">

**Mini-batch Gradient Descent**

<img width="1022" alt="Ảnh màn hình 2024-08-22 lúc 15 15 56" src="https://github.com/user-attachments/assets/86d24fad-71b7-495c-8212-c726acb65678">

- sự kết hợp giữa 2 pp trên.

sử dụng mini batch gradient decent. batchsize = 32 mẫu, N = 320 mẫu. cứ 1 lầm mô hình đc update gọi là 1 interation . vậy 1 epoch = ? iteration? 5 epoch = ? iteration 

1 epoch = 320/32 = 10  iteration -> 5 epoch = 50 iteration

## 4. Các thành phần của Neural Network

![Ảnh màn hình 2024-08-22 lúc 15 20 35](https://github.com/user-attachments/assets/f5101ad0-1033-4cef-b6cb-fae1a5e93986)
- dense layer = fully connected layer. Dense layer hay còn gọi là Fully connected layer là 1 lớp của mạng neural network. Mỗi dense layer gồm nhiều node gọi là các neural, mà trong đó mỗi neural sẽ nhận đầu vào là các neural thuộc lớp trước đó.

**Hiện tượng Gradient Vanishing là gì?**

Gradient vanishing xảy ra khi các gradient (đạo hàm của hàm mất mát theo các trọng số) trở nên rất nhỏ trong quá trình truyền ngược (backpropagation), đặc biệt là ở các lớp gần đầu vào của mạng. Khi gradient trở nên quá nhỏ, các trọng số của các lớp này không được cập nhật một cách đáng kể trong quá trình huấn luyện, dẫn đến mạng không học được hoặc học rất chậm.

**Các hàm kích hoạt thông dụng:**

- Công dụng của hàm kích hoạt:

Đưa sự phi tuyến tính vào mô hình

Giúp giới hạn output của neural

Góp phần hạn chế gradient vanishing (trái ngược với gradient vanishing là exploding gradient)

<img width="947" alt="Ảnh màn hình 2024-08-10 lúc 10 22 21" src="https://github.com/user-attachments/assets/4dbc70b7-1eae-481f-a15c-35621a3ba276">

**Một số nguyên tắc:**

- Trong bài toán Binary Classification, giả sử neural network tại lớp output chỉ có 1 neural => dùng sigmoid làm hàm kích hoạt (vì hàm này giới hạn output từ 0-1)

- Trong bài toán Multiclass classification, giả sử neural network tại lớp output có nhiều neural => dùng hàm softmaxlàm hàm kích hoạt.

**Công thức của hàm Softmax**

<img width="855" alt="Ảnh màn hình 2024-08-10 lúc 10 33 37" src="https://github.com/user-attachments/assets/b3859b7d-f256-46a1-9a89-f5e9221ed81e">

- có cách nào làm nhanh để tính đc mẫu này thuộc nhãn nào không -> Bạn không cần tính toàn bộ xác suất thông qua hàm Softmax, chỉ cần tìm giá trị lớn nhất trong vector đầu ra của lớp logits và xác định vị trí của nó. Nhãn tương ứng với vị trí này chính là nhãn của mẫu.

**Hàm lỗi (Loss function)**

- Hàm lỗi hay còn gọi là hàm mất mát (cost function) là 1 thành phần bắt buộc phải xác định trong mọi mạng neural network.

- Một số hàm lỗi thông dụng cho bài toán classification: binary cross entropy và categorical cross entropy.

<img width="1009" alt="Ảnh màn hình 2024-08-22 lúc 15 39 09" src="https://github.com/user-attachments/assets/f600c107-c6de-4fae-b8a7-c22b02ee3427">

**Optimizer**

- Optimizer (hàm tối ưu) là các hàm được dùng để tinh chỉnh trọng số của mô hình.

- Trong các phần trên, chúng ta đã tìm hiểu về thuật toán Gradient Descent, trên thực tế, ngoài Gradient Descent còn nhiều thuật toán tối ưu khác mà thư viện Pytorch hỗ trợ: Adam, Adagrad, RMSProp…

- khuyên dùng  thuật toán tối ưu Adam

**Hyperparameters vs Parameters**

- Parameters là từ chỉ chung các trọng số của mô hình mà chúng ta phải update

- Hyperparameters là từ chỉ các tham số phải được xác định trước khi quá trình training bắt đầu. (Ví dụ: Số lượng epoch, Batch size, Cấu trúc của mô hình (số lượng hidden layers, số lượng node trong 1 layer…), Các tham số của hàm mất mát, Learning rate…). Hyperparameters không update được mà phải dùng các phương pháp tunning để tìm ra bộ trọng số tốt nhất. 

# Buổi học 9: Neural Network (tt) (11/08/2024)

## DEMO: Pytorch MLP MNIST

https://www.kaggle.com/code/hongngcthuthng/pytorch-mlp-mnist?scriptVersionId=192013873

CHUẨN BỊ DATA

- Bước 1: tạo data set class để đọc data

Tạo data loader để tìm cách đưa data đi vào model 

- Bước 2: tạo ra class để đọc tập train

self.transform là biến lưu giữ các 1 pp augmentation mà chúng ta chọn (over sampling cho hình ảnh trong deeo learning)

- Bước 3: tạo ra class để đọc tập test

- Bước 4: Truyền đường dẫn

chuyển array thành dạng để cho pytorch đọc đc bằng câu lệnh: transform=transforms.ToTensor())

- Buơc 5: Chuẩn bị cách đọc data cho model

____

CẤU TRÚC MẠNG NEURAL NETWORK

<img width="491" alt="Ảnh màn hình 2024-08-11 lúc 09 38 30" src="https://github.com/user-attachments/assets/97a1ced9-73d6-469f-aeee-f82b0e2fc981">

<img width="493" alt="Ảnh màn hình 2024-08-11 lúc 09 38 38" src="https://github.com/user-attachments/assets/50c08390-301f-465a-a994-ab9fcdd863e4">

<img width="493" alt="Ảnh màn hình 2024-08-11 lúc 09 38 47" src="https://github.com/user-attachments/assets/c2e5f8bc-da45-43bf-b65b-83acfe7a3cf7">

# Buổi học 9: Một số Kĩ Thuật bổ sung (11/08/2024)

- Cross-validation là 1 pp đánh giá performance của mô hình

**Ưu điểm**

- có thể đánh giá trên toàn bộ dataset -> ưu việt hơn pp thông thường. 

- tránh được sampling bias

**Cách search ra bộ siêu tham số tốt nhất**

- Hyperclassifiers search

- pycaret -> kỹ thuật dành cho những ngừoi muốn lowcode khi làm machine learning

**Hyperclassifiers search**

- Cho phép thực hiện grid search CV hoặc random search CV trên nhiều câu lệnh cùng 1 lúc

**Pycaret**

https://pycaret.gitbook.io/docs

https://colab.research.google.com/drive/1C7AbOA7L1gfopuFzPtlG3yCXt1UAuIjz?usp=sharing&authuser=1

- low-code libery, đơn giản và dễ dùng hơn

# Buổi học 10: Unsupervised Learning – Cluster Analysis - KMeans (16/08/2024)

https://people.revoledu.com/kardi/tutorial/Clustering/Numerical%20Example.htm?authuser=1

## 1 Cluster Analysis 

**Ví dụ**

<img width="700" alt="Ảnh màn hình 2024-08-16 lúc 19 26 18" src="https://github.com/user-attachments/assets/ce1d4cfe-0582-4ef7-b0a0-8780f2aad8f7">

**Ghi chú: trong Cluster Analysis**

-  Không có khái niệm cluster "đúng"/sai, chỉ có phù hợp hơn

**Một số thuật toán cluster**

- K-Means clustering

- GMM

## 2. K-Means

**Điều kiện dừng**

- iterations = 100

- centroid << theshold (khi thay đổi quá ít thì sẽ tự động dừng**

**Trong trường hợp k có domain knowledge**

**Hard (Kmeans) vs Soft clusering (GMM)**

# Buổi học 11: Unsupervised Learning – Cluster Analysis - GMM (18/08/2024)

https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/

## 1. Dẫn nhập

## 2. Phân phối Gaussian

## 3. Mô hình Gaussian hỗn hợp (GMM)

## 4. Thuật toán EM (Kỳ vọng - Cực đại)

## 5. Ưu/khuyết điểm

## 6. Xây dựng GMM

# Buổi học 11: Unsupervised Learning - PCA (Principal ComponentAnalysis) (18/08/2024)

## 1. Dimensionality 

- hạn chế việc information loss nhất có thể khi giảm chiều dữ liệu

## 2. Reduction

## 3. Giới thiệu PCA

## 4. Ưu/khuyết điểm

## 5. Xây dựng PCA

# Buổi học 12: Unsupervised Learning - PCA (Principal ComponentAnalysis) (21/08/2024) (tt)

https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

## DEMO: Solution 

## DEMO 1: 
## What is shihoutte score?

# Buổi học 12: Time Series Forecasting - ARIMA (21/08/2024)

## 1. Giới thiệu

- time series là dữ liệu dạng chuỗi, ví dụ giá vàng, giá gạo,..

- ứng dụng vào bài toán như predictive maintainance

- điều kiện của data dạng time series:

data có yếu tố về thời gian, data dạng chuỗi (series/sequences), không nhất thiết phải ở dạng numberical mà còn là categorical, binary, hình,..

### ARIMA là viết tắt của Auto Regressive Integrated Moving Average**

ARIMA kết hợp ba khái niệm chính:

**AutoRegressive (AR) - Tự hồi quy**

- "tự động hồi quy". AutoRegressive (AR) là một cách để dự đoán giá trị tương lai của một chuỗi thời gian bằng cách sử dụng thông tin từ các giá trị trong quá khứ của chính chuỗi đó.

- ví dụ: Giả sử bạn muốn dự đoán nhiệt độ hôm nay dựa trên nhiệt độ của những ngày trước đó. Nếu bạn dùng nhiệt độ của ngày hôm qua để dự đoán nhiệt độ hôm nay, đó là một mô hình AR(1) (tự hồi quy bậc 1). Nếu bạn dùng nhiệt độ của hai ngày trước đó để dự đoán nhiệt độ hôm nay, đó là một mô hình AR(2).

**Integrated**

**Moving Average - Trung bình trượt**

- Moving Average (MA) là một mô hình dự báo trong đó giá trị hiện tại của chuỗi thời gian được xác định bằng cách sử dụng trung bình có trọng số của các sai số trong quá khứ.

### ARIMA có 2 loại: 

Để kiểm nghiệm xem data có tính seasonality không thì sẽ có phép thử chứ không cần thiết phải thực hiện theo cảm tính 

**ARIMA theo mùa (seasonal)**

- Được sử dụng khi chuỗi thời gian có các yếu tố mùa vụ rõ ràng. Ví dụ, khi bạn dự đoán doanh số bán hàng có chu kỳ tăng cao vào mùa lễ hội hoặc giảm mạnh vào mùa hè.

- Nếu mô hình có thành phần theo mùa, chúng ta sử dụng mô hình ARIMA theo mùa (SARIMA). Trong trường hợp đó, sẽ có một bộ tham số khác: P, Dvà Q mô tả các liên kết tương tự như p, d và q, nhưng tương ứng với các thành phần theo mùa của mô hình (Seasonal model)

**ARIMA không theo mùa (non-seasonal)**

- Được sử dụng khi chuỗi thời gian không có yếu tố mùa rõ rệt. Ví dụ, khi bạn dự đoán doanh số bán hàng không bị ảnh hưởng bởi các yếu tố mùa vụ.

## 2. Dự đoán - thuật toán

- Time Series có một số tính năng chính như xu hướng (trend), tính thời vụ (seasonality) và nhiễu (noise). (noise ở đây chính là những gì mà mô hình không thể giải thích được

- Công việc của chúng ta là phân tích các tính năng này của tập dữ liệu time series và sau đó áp dụng mô hình để d ự đoán trong tương lai.

- Trong mô hình ARIMA có 3 tham số được sử dụng để giúp mô hình hóa các khía cạnh chính của một chuỗi thời gian: seasonality, trend, và noise. Các tham số này được gắn nhãn lần lượt là p, d và q.

p là tham số kết hợp với khía cạnh tự động hồi quy của mô hình (auto-regressive aspect - AR)

d (difference): là tham số kết hợp với phần tích hợp của mô hình (integrated part- l)

q: là tham số liên quan đến phần trung bình động của mô hình (moving average part - MA)

**Dữ liệu theo stationary**

Stationary data (dữ liệu tĩnh) là một chuỗi thời gian có các đặc tính thống kê như trung bình, phương sai, và hiệp phương sai không thay đổi theo thời gian. Nói cách khác, một chuỗi thời gian được gọi là tĩnh khi nó không có xu hướng rõ ràng (trend), không có yếu tố mùa vụ, và các biến động của nó ổn định qua thời gian. Trong bài toán ARIMA phải chuyển dữ liệu về stationary. 

Đặc điểm của chuỗi thời gian tĩnh (Stationary):

- Mean (trung bình) của chuỗi không nên là một hàm theo thời gian. dữ liệu dạng stationary là dữ liệu có mean không thay đổi theo thời gian.

<img width="629" alt="Ảnh màn hình 2024-08-23 lúc 17 47 29" src="https://github.com/user-attachments/assets/1c803c57-52a9-4244-bac2-d57a9e72103e">

- Variance (phương sai)  của chuỗi không nên là một hàm theo thời gian.

<img width="593" alt="Ảnh màn hình 2024-08-23 lúc 17 49 01" src="https://github.com/user-attachments/assets/4a66dc2c-b49e-4d5a-af52-e07876a8bea0">

-  Covariance (hiệp phương sai) của thời gian thứ i và thời gian thứ (i +m) không nên là một hàm theo thời gian.

 <img width="593" alt="Ảnh màn hình 2024-08-23 lúc 17 50 17" src="https://github.com/user-attachments/assets/6e65d3cc-0f48-4445-83ee-6cf7e71c7572">

### Time series decomposition (phân tích thành phần chuỗi thời gian)

- là một kỹ thuật phân tích được sử dụng để tách một chuỗi thời gian thành các thành phần cơ bản để hiểu rõ hơn về các yếu tố ảnh hưởng đến chuỗi đó. Mục tiêu của phân tích này là để chia chuỗi thời gian thành các phần mà mỗi phần đại diện cho một yếu tố đặc trưng khác nhau của dữ liệu.

**Các thành phần cơ bản của Time Series Decomposition:**

- Trend

- Seasonality

- Cyclicity: Một chu kỳ xảy ra khi dữ liệu biểu hiện tăng và giảm không có tần số cố định. Những biến động này thường là do điều kiện kinh tế, và thường liên quan đến "business cycle". Thời gian của những biến động này thường tí nhất là 2 năm. (ví dụ: chu kì suy thoái kinh tế,..)

-  Residuals: độ lỗi trong dự đoán, những thứ mà time series không giải thích được
  
<img width="868" alt="Ảnh màn hình 2024-08-23 lúc 18 01 27" src="https://github.com/user-attachments/assets/2fddae97-95e2-49f1-b6f5-2e0f385f4007">

**Decomposition (phân tích)**

- đây là kĩ thuật được dùng để phân tích xem dữ liệu của chúng ta thuộc Seasonality hay Cyclicity.

- kỹ thuật này sẽ phân tích dữ liệu time series của chúng ta ra thành 3 phần: Trend (chuyển động lên hoặc xuống của đường cong dài hạn (long term), Seasonal component (thành phần theo mùa), Residuals

- mô hình cộng (additive model): sự thay đổi tuyến tính

- mô hình nhân (multiplicative model): sự thay đổi phi tuyến tính

<img width="869" alt="Ảnh màn hình 2024-08-23 lúc 18 10 31" src="https://github.com/user-attachments/assets/8d4c7f0a-2b34-49b3-9853-1effb5a85a3e">

## 3. Áp dụng auto_arima xây dựng mô hình ARIMA

**AIC (The Akaike information criterion):**

- Giá trị AIC cho phép so sánh mô hình phù hợp với  dữ liệu và tính đến độ phức tạp của mô hình, vì vậy các mô hình phù hợp hơn trong khi sử dụng ít tính năng hơn sẽ nhận được điểm AIC tốt hơn (thấp hơn) các mô hình tương tự sử dụng nhiều tính năng hơn
