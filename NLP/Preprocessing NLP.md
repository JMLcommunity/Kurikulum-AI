# Preprocessing Data pada NLP
Natural Language Processing (NLP) bertujuan untuk **memahami konteks dan arti** dari sebuah teks sehingga **memahami model bahasa** yang digunakan. <br>

Misal kita mempunyai data teks berupa dua kalimat yaitu

"Words are flowing out like endless rain into papercup" dan

"They slither while they pass, they slipp away across the universe"

(diambil dari [artikel ini](https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1)).

Maka untuk melakukan NLP pertama kita harus melakukan preprocessing teks dengan langkah-langkah sebagai berikut.

## Tokenisasi
Tokenisasi adalah proses **mensegmentasi teks menjadi kata**. Misal untuk data di atas maka akan menjadi :

<center><img src = "https://miro.medium.com/max/665/0*B1unFTAWyxHfhnN7"  height = "75" width = "550"></center>

## Stopwords Removal
Sesuai namanya, kita akan menghapus kata-kata yang menjadi *stopwords* seperti "the" dan "a".

## Stemming
Proses ini dilakukan untuk **menghilangkan imbuhan** yang ada dalam suatu kata semisal "playing" menjadi "play". Namun kita harus berhati-hati untuk tidak menghapus imbuhan yang bisa menghilangkan arti kata tersebut misalkan "news" menjadi "new". Untuk menghindari hal tersebut, terdapat banyak algoritma *stemming* yang telah disediakan.

## Lemmatization
Proses ini dilakukan untuk **menjadikan kata-kata yang mempunyai "*root*" yang sama menjadi *root*-nya**. Misalkan *went* menjadi *go* atau *best* menjadi *good*.

## Membuat Word Embedding
*Word Embedding* adalah vektor repsentasi dari setiap kata pada data teks (yang biasanya sudah di*stemming* atau di*lemmatisasi*). Beberapa teknik untuk melakukan mendapatkan *Word Embedding* adalah sebagai berikut : <br>

+ Bag of Words (BoW)<br>

  Ini adalah *Word Embedding* paling sederhana yaitu menghitung **frekuensi kemunculan setiap kata dalam sebuah teks**. Lihat contoh di bawah.
<br>
<br>
<center><img src="https://miro.medium.com/max/1325/0*myT5Z2GxTdJTUCsi" height = "100" width = "550"></center>
<br>
Teknik ini punya kelemahan : <b>tidak bisa memahami konteks</b>
<br>
<br>

+ Terms Frequency-Inverse Document Frequency (TF-IDF)<br>

  Teknik ini berusaha untuk memberi bobot pada tiap kata yang muncul. Terdapat dua proses yaitu *Term Frequency* (TF) dan *Inverse Document Frequency* (IDF). **TF adalah teknik untuk menghitung banyaknya kata di satu teks tertentu** (semakin banyak jumlahnya semakin besar TF / bobotnya), sementara **IDF akan mendiskon bobot dari TF bila kata tersebut muncul berkali-kali di teks lainnya** (lanjutkan baca [di sini](https://informatikalogi.com/term-weighting-tf-idf/))<br>
<br>

+ Word2Vec<br>

  Lebih lengkap tentang teori Word2Vec bisa dibaca [di sini](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)
