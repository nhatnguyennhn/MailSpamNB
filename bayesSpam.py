import numpy 
import pandas 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
duLieu = pandas.read_csv('train.csv', encoding='latin-1')[['v1', 'v2']] #thêm cái của nợ v1 v2 vì data nó 5 cột mà chỉ có cột v1 v2 @@ má tứk thằng tạo data ghê lun ak
duLieu.columns = ['label', 'mail']
NB = Pipeline([
    ('vectorizer', TfidfVectorizer()), 
    ('classifier', MultinomialNB())                  
])
x_train, x_test, y_train, y_test = train_test_split(duLieu['mail'], duLieu['label'], test_size=0.20, random_state = 21) #không có dữ liệu để test nên lấy bừa 1 số cái trong train để test
NB.fit(x_train, y_train)

duDoan=NB.predict(x_test)
ketqua = pandas.DataFrame({
    'du doan': duDoan,
    'mail': x_test
})

ketqua.to_csv('ketqua.csv', index = False) #sau khi đối chiếu kết quả và tính score thì tỉ lệ chính xác  97%

