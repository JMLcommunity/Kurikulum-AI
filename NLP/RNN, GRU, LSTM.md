# RNN, GRU, dan LSTM

<div style = "text-align: justify ; line-height: 1.75 em"><b>RNN</b> adalah modifikasi dari <i>feedforward neural network</i> yang mempunyai <b>memori internal</b> dan memori ini akan <b>dipanggil</b> dalam proses </i>training</i> di <b>neuron <i>input</i> selanjutnya</b>. Ini menyebabkan struktur RNN cocok digunakan dalam pengolahan data berupa <b><i>sequence</i></b> seperti kalimat atau <i>time series</i>. Perhatikan skema di bawah (dari <a src = "http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/">situs ini</a>).<br>
<br>
<center><img src = "http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg" height = "125" width = "500"></center>
<br>
Perhatikan bahwa X<sub>t</sub> dan s<sub>t-1</sub> akan menjadi input untuk menghasilkan s<sub>t</sub> dan seterusnya atau dapat dituliskan s<sub>t</sub> = f(Ws<sub>t-1</sub>,Ux<sub>t</sub>). Selanjutnya f bisa berupa fungsi aktivasi apapun misalkan tanh, sigmoid, atau ReLU dan neuron <i>output</i> pada saat t dapat dituliskan (misalkan fungsi yang dipakai adalah softmax) dalam o_pred<sub>t</sub> = softmax(Vs<sub>t</sub>).<br>
RNN mempunyai beberapa kekurangan diantaranya adalah <i>vanishing gradient problem</i>. Ini dapat diilustrasikan sebagai berikut.<br>
Misalkan </i>loss function</i> yang digunakan adalah <i>cross entropy</i> sehingga dapat dituliskan dalam bentuk <br>
E<sub>t</sub> = -o<sub>t</sub> $\log$ o_pred<sub>t</sub><br>
E = $\Sigma_t$-o<sub>t</sub> $\log$ o_pred<sub>t</sub><br>
Tujuan kita adalah mencari gradien (turunan) dari <i>loss function</i> terhadap U,V, dan W sehingga didapat U, V, dan W paling baik. Perhatikan bahwa saat kita ingin mencari W paling optimal maka haruslah dicari
<br>
<br>
<center><img src = "http://s0.wp.com/latex.php?zoom=1.25&latex=%5Cfrac%7B%5Cpartial+E%7D%7B%5Cpartial+W%7D+%3D+%5Csum%5Climits_%7Bt%7D+%5Cfrac%7B%5Cpartial+E_t%7D%7B%5Cpartial+W%7D&bg=ffffff&fg=000&s=1"></center><br>
Misalkan untuk t = 3, maka dengan <i>chain rule</i> didapat
<br>
<br>
<center><img src = "http://s0.wp.com/latex.php?zoom=1.25&latex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+E_3%7D%7B%5Cpartial+W%7D+%26%3D+%5Cfrac%7B%5Cpartial+E_3%7D%7B%5Cpartial+%5Chat%7By%7D_3%7D%5Cfrac%7B%5Cpartial%5Chat%7By%7D_3%7D%7B%5Cpartial+s_3%7D%5Cfrac%7B%5Cpartial+s_3%7D%7B%5Cpartial+W%7D%5C%5C++%5Cend%7Baligned%7D++&bg=ffffff&fg=000&s=0"></center><br>
Namun perhatikan bahwa s<sub>3</sub> = tanh(Ux<sub>t</sub> + Ws<sub>2</sub>) bergantung pada s<sub>2</sub> dimana juga bergantung pada s<sub>1</sub>. Akhirnya akan didapat <br><br>
<center><img src = "http://s0.wp.com/latex.php?zoom=1.25&latex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+E_3%7D%7B%5Cpartial+W%7D+%26%3D+%5Csum%5Climits_%7Bk%3D0%7D%5E%7B3%7D+%5Cfrac%7B%5Cpartial+E_3%7D%7B%5Cpartial+%5Chat%7By%7D_3%7D%5Cfrac%7B%5Cpartial%5Chat%7By%7D_3%7D%7B%5Cpartial+s_3%7D%5Cfrac%7B%5Cpartial+s_3%7D%7B%5Cpartial+s_k%7D%5Cfrac%7B%5Cpartial+s_k%7D%7B%5Cpartial+W%7D%5C%5C++%5Cend%7Baligned%7D++&bg=ffffff&fg=000&s=0"></center><br>
Namun belum berhenti sampai di sana, perhatikan bahwa
<br>
<br>
<center><img src = "http://s0.wp.com/latex.php?zoom=1.25&latex=%5Cfrac%7B%5Cpartial+s_3%7D%7B%5Cpartial+s_1%7D+%3D%5Cfrac%7B%5Cpartial+s_3%7D%7B%5Cpartial+s_2%7D%5Cfrac%7B%5Cpartial+s_2%7D%7B%5Cpartial+s_1%7D&bg=ffffff&fg=000&s=1"></center><br>
sehingga menghasilkan
<br>
<br>
<center><img src = "http://s0.wp.com/latex.php?zoom=1.25&latex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+E_3%7D%7B%5Cpartial+W%7D+%26%3D+%5Csum%5Climits_%7Bk%3D0%7D%5E%7B3%7D+%5Cfrac%7B%5Cpartial+E_3%7D%7B%5Cpartial+%5Chat%7By%7D_3%7D%5Cfrac%7B%5Cpartial%5Chat%7By%7D_3%7D%7B%5Cpartial+s_3%7D++%5Cleft%28%5Cprod%5Climits_%7Bj%3Dk%2B1%7D%5E%7B3%7D++%5Cfrac%7B%5Cpartial+s_j%7D%7B%5Cpartial+s_%7Bj-1%7D%7D%5Cright%29++%5Cfrac%7B%5Cpartial+s_k%7D%7B%5Cpartial+W%7D%5C%5C++%5Cend%7Baligned%7D++&bg=ffffff&fg=000&s=0"></center><br>
Jadi kita harus mencari gradien dari fungsi tanh untuk setiap <i>time stamp</i> dan mengalikannya. Padahal bila dilihat dari grafik fungsi tanh dan turunannya di bawah ini
<br>
<br>
<center><img src = "https://nn.readthedocs.io/en/rtd/image/tanh.png" height= "450" width = "450"></center><br>
dapat diambil kesimpulan bahwa semakin besar atau kecil input dari fungsi tanh maka hasilnya akan mendekati 1 atau -1 sehingga gradiennya mendekati 0. Maka <b>semakin jauh jarak suatu s dari s<sub>t</sub> (belum dikenai fungsi tanh berkali-kali), hasil perkalian turunan s terhadap semua s sebelumnya mendekati nol sehingga kontribusi input di suatu neuron yang jaraknya jauh tersebut untuk keseluruhan gradien <i>loss function</i> terhadap W dapat diabaikan</b>. Artinya, kita tidak akan bisa menentukan <i>output</i> berdasarkan <i>input</i> yang diingat untuk jangka waktu yang panjang.<br>
Untuk mengatasi hal ini maka dikembangkanlah <i>Long Short Term Memory</i> (LSTM) dan <i>Gated Recurrent Unit</i> (GRU). LSTM mempunyai arsitektur yang bisa menyeleksi mana memori lama yang bisa ditambahkan, mana yang harus dilupakan, dan mana yang bisa ditambahkan ke memori selanjutnya. Berikut adalah arsitektur dari LSTM (diambil dari <a src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/">sini</a>)
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" height = "200" width = "500"></center>
<br>
Core dari arsitektur tersebut adalah <b><i>cell state</i></b> yaitu struktur lurus yang berada di atas struktur memori LSTM ini.
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png" height = "200" width = "550"></center>
<br>
Informasi bisa mengalir di <i>cell state</i>, ditambahkan, ataupun dikeluarkan. Untuk menyeleksi mana informasi yang bisa diperbolehkan bercampur dengan informasi di <i>cell state</i>, digunakanlah sistem <b><i>gate</i></b> yang terdiri dari neuron dengan fungsi aktivasi sigmoid dan operator perkalian <i>point-wise</i>. Fungsi sigmoid ini memetakan input ke antara 0 dan 1 (0 berarti semua informasi tidak boleh masuk dan 1 sebaliknya).
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png" height = "150" width = "150"></center>
<br>
Pertama kita informasi lama akan masuk lewat <i>cell state</i> dan terdapat kombinasi <i>hidden state</i> lama dan baru yang masuk lewat <i>forget gate</i> untuk diseleksi oleh fungsi sigmoid apakah info lama tersebut akan dilupakan (pengalinya kecil) atau tidak.
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" height = "200" width = "550"></center>
<br>
<br>
Setelah itu kita juga akan menentukan sebesar apa gabungan informasi di <i>state</i> yang dulu dan sekarang dapat berpengaruh ke <i>cell state</i> dengan fungsi aktivasi sigmoid dan apakah pengaruhnya positif atau negatif dengan fungsi aktivasi tanh.
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" height = "200" width = "550"></center>
<br>
Sehingga hasilnya informasi <i>state</i> yang dulu (setelah melalui proses pemilihan untuk dilupakan atau tidak) dapat <b>ditambahkan</b> dengan kombinasi <i>state</i> dahulu dan sekarang. Perhatikan bahwa karena <b>informasi lama ditambahkan, maka bisa lebih rentan terhadap <i>vanishing gradient problem</i></b> namun juga lebih bagus karena informasi lama yang ditambahkan sudah diseleksi terlebih dahulu. Bila diputuskan untuk tidak berkontribusi, input tersebut memang tidak relevan dengan <i>state</i> saat ini dan bukan karena terlalu jauh jaraknya dari <i>state</i> saat ini.<br>
Akhirnya akan dikeluarkanlah output C<sub>t</sub> yang masuk ke <i>cell state</i> selanjutnya. Namun <b>seberapa kombinasi output dari <i>state</i> saat ini untuk mengeluarkan <i>hidden layer</i> juga akan diseleksi dengan <i>output gate</i></b> dengan sistem di bawah ini.
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" height = "200" width = "500"></center>
<br>
<br>
Itulah dasar cara kerja LSTM. LSTM sendiri punya beberapa variansi seperti GRU yang menggabungkan <i>forget</i> serta <i>input gate</i> jadi satu serta <i>cell state</i> dengan <i>hidden layer</i> seperti gambar di bawah ini.
<br>
<br>
<center><img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png" height = "200" width = "550"></center>
</div>
