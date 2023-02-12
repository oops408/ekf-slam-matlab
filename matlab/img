I = imread('image.jpg');
I_gray = rgb2gray(I);
I_smooth = imgaussfilt(I_gray, 2);
I_edges = edge(I_smooth, 'Canny');
figure;
subplot(1,2,1);
imshow(I);
title('Original Image');

subplot(1,2,2);
imshow(I_edges);
title('Edge Detection Result');
