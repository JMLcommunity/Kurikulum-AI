### Teori : Transformer, BERT, serta variansi dari BERT

<div style = "text-align: justify ; line-height: 1.75 em">RNN dan LSTM masih punya beberapa kelemahan utama yaitu <b>pengolahan dilakukan <i>sequence by sequence</i></b> dan <b>masih berpaku pada <i>long term dependency</i></b>. Pemrosesan <i>sequence to sequence</i> sangat lama dan menyebabkan paralesasi dengan GPU tidak dapat dilakukan. Sementara <i>long term dependency</i> menyebabkan pengolahan teks yang semakin banyak menjadi tidak efektif. Maka dari itu semua <i>long term dependency</i> dan proses <i>recurrent</i> dihilangkan pada arsitektur Transformer. Prinsip dari Transformer adalah <b><i>Attention is All You Need</i></b> dengan bentuk arsitektur seperti berikut.
<br>
<br>
<center><img src = "https://miro.medium.com/max/1898/1*HKyS_RuocFun1LUubSf-jQ.jpeg" height = "600" width = "600"></center><br>
Secara singkat, kita <b>menjadikan input dan output menjadi vektor angka dengan teknik <i>word embedding</i> </b>(baca <a src = "https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa">di sini</a>) agar teks bisa diolah oleh neuron serta dengan teknik kedua yaitu <b><i>positional encoding</i></b> dengan fungsi sin dan cos (mengapa sin cos? baca <a src = "https://kazemnejad.com/blog/transformer_architecture_positional_encoding/">di sini</a>) agar posisi suatu kata dapat diingat. Kedua teknik ini menciptakan <i>embedding with time signal</i> yang dapat dianggap konteks terhadap suatu kata pada suatu teks dan semua proses ini berlangsung sekaligus. Selanjutnya input akan masuk ke <b><i>Encoder</i></b> dan output akan masuk ke <b><i>Decoder</i></b>. Terdapat beberapa layer <i>Encoder</i> dan <i>Decoder</i> yang harus dilewati tapi struktur nya sama : <b><i>Multi-Head Attention + Fully Connected Layer</i></b> untuk <i>Encoder</i> dan <b><i>Multi-Head Attention + Encoder-Decoder Attention + Fully Connected Layer</i></b> untuk <i>Decoder</i> (silahkan baca <a src = "http://jalammar.github.io/illustrated-transformer/">di sini</a> untuk mekanisme yang sangat jelas tentang mekanisme <i>attention</i>). <br>
Intinya, untuk bagian <b><i>encoder</i></b> masing-masing <b><i>embedding</i> dengan <i>time-stamp</i></b> akan <b>dikalikan dengan matriks <i>weight</i></b> yang di<i>train</i> selama proses <i>training</i> sehingga menghasilkan matriks bernama <b>Query, Key, dan Value</b>. Ketiga matriks ini dimasukkan ke fungsi softmax dengan persamaan seperti di bawah ini.
<br>
<br>
<center><img src = "http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" height = "200" width = "400"></center>
<br>
<br>
Proses ini dilakukan berulang-ulang dan matriks Z yang didapat nantinya dikonkatenasi dan dikalikan dengan matriks <b>weight</b> tertentu lagi sehingga menghasilkan mekanisme <b><i>Multi Headed Attention</i></b>. <i>Multi Headed Attention</i> dapat ditunjukkan oleh gambar berikut.
<br>
<br>
<center><img src = "http://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png" height = "550" width = "550"></center>
<br>
Dapat dilihat bahwa untuk kata-kata "it" maka Transformer akan memberikan perhatian lebih pada "The animal" dan "tired" yang mana ini tidak dapat dilakukan bila kalimat tersebut diproses <i>sequence by sequence</i>.
Hal yang perlu diperhatikan juga dalam arsitektur ini, setiap output dari mekanisme <i>attention</i> ataupun <i>fully connected layer</i> akan ditambah vektor inputnya dan dilakukan <i>layer normalization</i> seperti gambar di bawah.
<br>
<br>
<center><img src = "http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" height = "500" width = "500"></center>
<br>
<br>
Untuk bagian <i>Decoder</i> mirip dengan <i>Encoder</i> hanya saja <b>input untuk bagian <i>Encoder Decoder Attention</i> sedikit berbeda</b>. <b>Q berasal dari semua hasil prediksi yang tersedia sementara K dan V berasal dari output <i>Encoder</i></b>.
<br>
<br>
<img src = "https://miro.medium.com/max/2031/1*in3wK8ex2fZJLsoZgKCQAQ.jpeg" height = "200" width = "550">
<br>
<br>
Untuk lebih lengkapnya perhatikan ilustrasi di bawah (diambil dari <a src = "http://jalammar.github.io/illustrated-transformer/">sini</a>)
<br>
<br>
<img src = "http://jalammar.github.io/images/t/transformer_decoding_1.gif" height = "450" width = "500">
<br>
<br>
<img src = "http://jalammar.github.io/images/t/transformer_decoding_2.gif" height = "450" width = "500">
<br>
<br>
Ternyata <b>Transformer</b> pun masih punya beberapa <b>kelemahan</b> dikarenakan <b>inputnya yang berupa <i>word embedding</i></b>. Kelemahan pertama adalah tidak bisa mempelajari konteks dari kata-kata tetangganya. Sementara kelemahan kedua adalah dibutuhkannya sebuah model yang general dapat dipakai untuk segala tipe kasus NLP seperti pada kasus di sini, penilaian Q&A. Maka dari itu dibuatlah sebuah <b>teknik <i>training</i></b> berbeda namun tetap menggunakan prinsip arsitektur Transformer yang dinamakan dengan BERT.<br>
Secara umum, urutan teknik <i>training</i> pada BERT adalah : (1) membuat <i>pretrained model</i> dari dua kasus NLP yaitu <i>Masked Language Model</i> (MLM) dan <i>Next Sentence Prediction</i> (NSP) sehingga menghasilkan <i>embedding</i> baru untuk input yang ada (dimana <i>embedding</i> ini <i>embedding</i> yang optimal untuk kasus MLM dan NSP) dan (2) menggunakan hasil <i>embedding</i> ini untuk dimasukkan lagi ke <i>neural network</i> yang dihubungkan dengan output sesuai kasus NLP yang diinginkan.
<br>
Bila kita telisik lebih jauh, input untuk BERT tidak harus satu kalimat namun bisa terdiri dari beberapa bagian teks untuk mendukung kasus NLP yang beragam. Misalkan terdapat 3 bagian dalam teks yaitu judul, pertanyaan, dan isi. Maka kita bisa memasukkannya secara bersamaan namun tetap terpisah ke dalam BERT. Perhatikan gambar di bawah (diambil dari <a src = "https://miro.medium.com/max/2000/1*D0_sVWpmOSaGCvm6gk9aHA.jpeg">sini</a>)
<br>
<br>
<center><img src = "https://miro.medium.com/max/2000/1*D0_sVWpmOSaGCvm6gk9aHA.jpeg" height = "200" width = "550"></center>
<br>
Dapat dilihat bahwa <b>input</b> kita terdiri dari <b>CLS + kalimat bagian pertama (misalkan judul) + SEP + kalimat bagian kedua (misalkan isi) + SEP + ...</b>. Input ini selanjutnya akan <b>ditokenisasi</b> dengan tokenisasi yang ada untuk BERT. Perhatikan bahwa <b>tokenisasi ini bahkan mentokenisasi imbuhan menjadi satu bagian sendiri</b>. Setelah itu dikalikan dengan suatu matriks sehingga menjadi <i>embedding</i> dan ditambahkan dengan <i>Segment Embedding</i> serta <i>Position Embedding</i>.
<br>
Langkah selanjutnya adalah melakukan <i>pre-training</i> untuk mendapat <b><i>vector representation</i></b> setiap kata menggunakan arsitektur <b>Multi Headed Attention dari Transformer</b>. Ingat bahwa dalam kasus <b>klasifikasi kalimat</b>, <b>hanya C yang merupakan output dari <i>embedding</i> CLS yang digunakan untuk klasifikasi sementara kasus lainnya memakai semua <i>embedding</i> kecuali C</b>.
<br>
<br>
<center><img src = "https://miro.medium.com/max/1835/1*zlNaJtkkpg2UaKl7ibKOyQ.png" height = "550" width = "550"></center>
<center><img src = "https://miro.medium.com/max/1568/1*UvFUs9afyoIGKj9F5qTIxw.png" height = "550" width = "550"></center>
<br>
<br>
Setelah didapat <b><i>vector representation</i></b> untuk setiap kata, baru vektor tersebut jadi input untuk melakukan tugas NLP tertentu. Di internet telah terdapat model BERT yang telah di<i>pretrain</i> yaitu Base dan Large. Perlu diperhatikan juga bahwa maksimal panjang kata yang dapat dimasukkan ke BERT adalah 512 kata dan nilai hyperparameter yang biasa dipakai adalah Dropout 0.1, Batch Size 16 / 32, LR Adam 5e-5, 3e-5, 2e-5, dan Epoch 3 - 5.
</div>
