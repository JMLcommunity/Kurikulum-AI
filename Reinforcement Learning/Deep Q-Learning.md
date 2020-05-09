## Prinsip Deep Q-Learning

<div style = "text-align: justify ; line-height: 1.75 em">Telah kita ketahui bahwa salah satu kelemahan <i>Q-learning</i> adalah memori dan waktu karena harus mempelajari dan menyimpan <i>Q-value</i> dari setiap <i>state</i> dan <i>action</i>. Bila kita batasi proses pembelajaran agent tersebut untuk menghemat waktu dan memori, maka tidak semua kemungkinan dicoba. Saat agent tersebut selesai belajar dan dijalankan terdapat kemungkinan agent tersebut menemukan <i>state</i> baru. Akibatnya agent tersebut tidak tahu harus berbuat apa. Dengan kata lain, kekurangan <i>Q-learning</i> adalah tidak melakukan generalisasi terhadap <i>state</i> dan <i>action</i> yang mungkin.
<br>
<br>
Namun bagaimana bila kita bisa mengestimasi berbagia kemungkinan <i>Q-value</i>? Itulah yang dilakukan oleh <i>Deep Q-learning</i>. <i>Deep Q-learning</i> berusaha mengestimasi <i>Q-value</i> dari <i>action</i> yang diambil untuk setiap <i>state</i> yang ada. Input dari <i>Deep Q-learning</i> adalah gambar <i>state</i> saat ini.
<br>
<br>
<center><img src = "https://pic4.zhimg.com/80/v2-67ef75bb7f5e67b2a42645aa821894bf_hd.png"></center>
<center>Ilustrasi Deep Q-learning pada game Atari (sumber: https://zhuanlan.zhihu.com/p/25239682)</center>
<br>
<b>Target dari proses <i>training neural network</i> ini adalah <i>immadiate reward</i> + ($\gamma$ x <i>estimate of future value</i>)</b> atau suku yang dinamakan <i>learned value</i> pada persamaan <i>update Q-value</i> dan <b><i>Loss function</i></b> yang ingin diminalkan adalah <b>(<i>Q-value</i> prediksi - <i>Q-value</i> sebenarnya)<sup>2</sup></b>. Tapi masalahnya adalah, kita juga tidak tau berapa <i>Q-value</i> sebenarnya saat mengambil <i>action</i> tertentu di <i>state</i> tertentu. Jadi kita harus memprediksi kedua nilai <i>Q-value</i> tersebut sehingga tidak mungkin digunakan satu neural network saja.
<br>
<br>
Untuk mengatasi masalah di atas, maka <b>satu input akan dimasukkan ke dua <i>neural network</i> yang berbeda</b>. Satu <i>neural network</i> berfungsi untuk mengestimasi target, dan <i>neural network</i> lainnya digunakan untuk memprediksi <i>Q-value</i> hasil prediksi. Lalu hasilnya training keduanya dimasukkan ke dalam <i>cost function</i>. Namun, <i>weight</i> yang selalu diupdate hanya <i>weight</i> dari <i>neural network</i> kedua sementara <i>weight</i> dari <i>neural network</i> pertama dibuat semi-konstan. Dengan kata lain, <i>weight neural network</i> pertama diperbarui dengan nilai <i>weight neural network</i> kedua hanya setiap beberapa iterasi sekali. Ini dilakukan terus menerus hingga <i>cost function</i> mencapai batas <i>threshold</i> minimal. Arsitektur dari <i>Deep Q-learning</i> dapat dilihat seperti gambar di bawah ini.
<br>
<br>
<img src = "https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/04/Screenshot-2019-04-17-at-12.48.05-PM-768x638.png" height = "500" width = "500">
<br>
<br>
Setalah mengetahui arsitekturnya, kita juga harus menentukan bagaimana cara <i>trainingnya</i> agar tidak menghabiskan memori dan waktu untuk melakukan training <i>neural network</i> sepanjang waktu. <b>Cara training</b> yang digunakan adalah <b><i>Experience Replay</i></b>. Jadi pertama kita tentukan berapa banyak <i>batch</i> yang akan dimasukkan ke <i>neural network</i> untuk di<i>train</i>. Kemudian jalankan terlebih dahulu agent untuk mengambil <i>Q-value</i> pada beberapa <i>state</i> dan <i>action</i> dan simpan dalam sebuah memori. Bila <i>Q-value</i> untuk setiap transisi <i>state</i> telah mencapai nilai <i>batch</i>, maka ambil data-data tersebut dan <i>train</i> di <i>neural network</i> dengan arsitektur seperti di atas. Selanjutnya jalankan kembali agent. Setiap terdapat data baru sejumlah <i>batch</i>, maka kita lakukan sampling sebanyak <i>batch</i> dan lakukan <i>training</i> di <i>neural network</i>. Keseluruhan workflow <i>Deep Q-learning</i> dapat dilihat di bawah (diambil dari https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287).
<br>
<br>
<img src = "https://miro.medium.com/max/1508/1*nb61CxDTTAWR1EJnbCl1cA.png" height = "500" width = "500">
</div>
