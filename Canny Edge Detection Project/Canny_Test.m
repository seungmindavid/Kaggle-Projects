cannyTest1 = imread('cannytest1.bmp');
cannyTest2 = imread('cannytest2.bmp');
cannyTest3 = imread('cannytest3.bmp');
cannyTest4 = imread('cannytest4.bmp');

sigma = 1;
threshold = [25:75];
connectivity_4 = 4;
connectivity_8 = 8;

imgResult1 = Canny(cannyTest1, sigma, threshold, connectivity_4);
imgResult2 = Canny(cannyTest2, sigma, threshold, connectivity_4);
imgResult3 = Canny(cannyTest3, sigma, threshold, connectivity_4);
imgResult4 = Canny(cannyTest4, sigma, threshold, connectivity_4);

imgResult5 = Canny(cannyTest1, sigma, threshold, connectivity_8);
imgResult6 = Canny(cannyTest2, sigma, threshold, connectivity_8);
imgResult7 = Canny(cannyTest3, sigma, threshold, connectivity_8);
imgResult8 = Canny(cannyTest4, sigma, threshold, connectivity_8);

subplot(2, 4, 1), imshow(uint8(imgResult1)), title("4 connectivity for cannytest1.bmp");
subplot(2, 4, 2), imshow(uint8(imgResult2)), title("4 connectivity for cannytest2.bmp");
subplot(2, 4, 3), imshow(uint8(imgResult3)), title("4 connectivity for cannytest3.bmp");
subplot(2, 4, 4), imshow(uint8(imgResult4)), title("4 connectivity for cannytest4.bmp");

subplot(2, 4, 5), imshow(uint8(imgResult5)), title("8 connectivity for cannytest1.bmp");
subplot(2, 4, 6), imshow(uint8(imgResult6)), title("8 connectivity for cannytest2.bmp");
subplot(2, 4, 7), imshow(uint8(imgResult7)), title("8 connectivity for cannytest3.bmp");
subplot(2, 4, 8), imshow(uint8(imgResult8)), title("8 connectivity for cannytest4.bmp");
