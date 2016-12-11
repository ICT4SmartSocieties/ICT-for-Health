close all
clear all
%Lab 8 - Moles - Image Processing 

A = imread('Images/low_risk_2.jpg');
% imshow(A)
[N1,N2,N3] = size(A);
N = N1*N2;
B = double(reshape(A,N,N3));