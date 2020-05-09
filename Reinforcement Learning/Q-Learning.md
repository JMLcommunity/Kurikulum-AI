# Reinforcement Learning dan Q-Learning

<div style = "text-align: justify ; line-height: 1.75 em"><b><i>Reinforcement Learning</b></i> adalah sebuah metode <i>machine learning</i> dimana kita <b>melatih sebuah agent</b> untuk belajar mengambil keputusan <b><i>action</b></i>) dalam sebuah kondisi tertentu (<b><i>state</b></i>) di lingkungan tertentu (<b><i>environment</b></i>) sehingga bisa memaksimalkan <b><i>reward</b></i> yang akan didapat (<i>reward</i> bisa bernilai positif atau negatif dan akan didapat bila berhasil melakukan tujuan yang diinginkan). Proses pembelajaran ini dilakukan pada setiap <b><i>episode</b></i> pembelajaran dimana <b><i>episode</b></i> adalah setiap kejadian diantara <i>initial state</i> (kondisi awal) dan <i>terminal state</i> (kondisi akhir). Pada <i>Reinforcement Learning</i>, diasumsikan bahwa kondisi yang akan datang hanya dipengaruhi oleh kondisi saat ini dan bukan kondisi-kondisi sebelumnya. Proses dengan asumsi seperti ini disebut juga Proses Markov</div>
<br>
<div style = "text-align: justify ; line-height: 1.75 em">Pada setiap prosesnya (setiap <i>state</i>), agent akan melakukan <i>action</i> untuk memaksimalkan <i>reward</i>. Maka kualitas (<b><i>quality</b></i>) dari setiap aksi dinilai dari seberapa besarnya total <i>reward</i> yang didapat. Total <i>reward</i> ini disebut juga dengan <b><i>Q-value</b></i>. Pada tiap <i>episode</i>, agent dilatih untuk berusaha meningkatkan <i>Q-value</i> di <i>state</i> tersebut. Pada setiap pembelajaran, maka akan didapat <i>Q-value</i> akan diperbarui menjadi :</div>

![Proses perbaruan *Q-value*](https://wikimedia.org/api/rest_v1/media/math/render/svg/47fa1e5cf8cf75996a777c11c7b9445dc96d4637)

dengan :

$\alpha$ = porsi seberapa diperhitungkannya pengetahuan baru menggantikan pengetahuan lama <br>
$\gamma$ = porsi seberapa diperhitungkannya reward yang didapat setelah <i>state</i> saat ini (bila 0 maka hanya <i>immadiate reward</i> yang diperhitungkan) <br>

<div style = "text-align: justify ; line-height: 1.75 em">Nantinya <i>Q-value</i> ini akan dimasukkan ke dalam <i>Q-table</i> yang berisi daftar <i>reward</i> untuk setiap <i>action</i> pada setiap <i>state</i>. Untuk lebih jelasnya maka perhatikan ilustrasi di bawah ini (diambil dari <a src = "https://towardsdatascience.com/q-learning-54b841f3f9e4">artikel ini</a>).
<br>
<br>
<center><img src = "https://miro.medium.com/max/750/1*tSFotpgBNGurajFg2FH8Cg.png"></center>
<br>
Pada sebuah permainan dengan misi mencari jalan terpendek dari 1,1 ke 5,5, terdapat <i>reward</i> yang ditunjukkan dengan warna hijau. Sementara warna merah adalah <i>punishment</i> (<i>reward</i> bernilai negatif sehingga mengurangi <i>Q-value</i>). Maka <i>Q-table</i> yang dihasilkan adalah
<br>
<br>
<center><img src = "https://miro.medium.com/max/981/1*p6yPonqoDMlK1w_EJKlcAQ.png"></center>
<br>
Artinya, pada titik 1,1, bila kita melakukan aksi bergerak ke atas atau ke kiri kita mendapat <i>punishment</i> sebesar -1000 sementara bila bergerak ke bawah atau ke kanan kita mendapat <i>reward</i> sebesar 1 dan seterusnya. Namun ini baru sistem yang hanya memperhatikan <i>immadiate reward</i> dan sistem ini tidak sempurna. Pada konsisi tertentu, hanya memperhitungkan <i>immadiate reward</i> ($\gamma$ = 0) punya beberapa kelemahan. Misalkan pada suatu <i>environment</i> yang berbeda yaitu permainan yang sama namun dengan suatu halangan, kita ada di titik 2.2
<br>
<br>
<center><img src = "https://miro.medium.com/max/750/1*XV1aCvN2kWkTaos-E1h1yQ.png"></center>
<br>
Maka terdapat 2 pilihan yaitu mengambil <i>immadiate reward</i> yang besar yaitu ke bawah atau mengambil <i>immadiate reward</i> yang lebih kecil dan keluar dari jalan buntu. Hal lainnya yang perlu diperhatikan adalah agent tersebut belum memperhitungkan banyaknya langkah yang harus diambil untuk memaksimalkan <i>Q-value</i>. Bisa saja untuk memaksimalkan <i>Q-value</i> maka agent kita melakukan <i>infinite looping</i> di daerah berwarna hijau dan mengambil <i>reward</i> sebanyak-banyaknya. Selain itu, bila kita perhatikan persamaan matematika di atas, maka agent akan terus mengambil <i>action</i> yang sama pada tiap <i>episode</i> latihan yang membuat <i>Q-value</i> sebesar-besarnya tanpa mengeksplorasi hal baru. Padahal <b>terdapat kemungkinan bahwa solusi terbaik adalah solusi yang belum dicoba</b>. Untuk mengatasi hal tersebut, kita harus membuat parameter lain yaitu <b>$\epsilon$</b> sehingga <b>bila <i>reward</i> yang didapat kurang dari atau sama dengan $\epsilon$</b>, kita buat agent tersebut <b>memilih <i>action</i> secara random</b>.

Itulah penjelasan tentang <i>Reinforcement Learning</i>, <i>Q-learning</i>, dan beberapa prosedurnya. Dapat dilihat bahwa <i>Q-learning</i> tidak menggunakan model apapun dan hanya berfokus pada <i>action</i> dan <i>reward</i> sehingga disebut juga <i>model-free</i>. Perlu diperhatikan bahwa <b>bila kemungkinan <i>state</i> serta <i>action</i> yang ada sangat banyak</b> maka kelemahan Q-Learning adalah harus <b>mendaftar satu persatu <i>Q-value</i></b> tersebut. Pastinya ini akan memakan banyak memori dan waktu setiap kali kita mengeksplorasi <i>state</i> dan <i>action</i> baru.
</div>
