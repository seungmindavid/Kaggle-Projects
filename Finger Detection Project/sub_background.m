hand = imread('handdd.JPG'); 
back = imread('bacc.JPG');

diff = abs(hand - back) > 10;
diff = rgb2gray(255 * uint8(diff));

% median filtering
med_hand = medfilt2(diff);
gau_hand = imgaussfilt(diff,2);
% graythresh can be used to compute the level argument automatically
binary_hand = im2bw(med_hand,graythresh(med_hand));

% Find the connected components
connected_component = bwareaopen(binary_hand, 550500);
Properties = regionprops(connected_component,'all'); 

% Looking for the maximum 
area = [Properties.Area];
index = find(area==max(area));
only_hand = Properties(index).FilledImage;
hand_convex = Properties(index).ConvexImage;

hand = insertMarker(hand, Properties(index).ConvexHull, 'o', 'size', 100);

edge = edge(hand_convex, 'canny');

fuse = imfuse(only_hand, edge);
subplot(3,2,1), imshow(only_hand)
subplot(3,2,2), imshow(hand_convex)
subplot(3,2,3), imshow(edge)
subplot(3,2,4), imshow(fuse)
subplot(3,2,5), imshow(hand)

