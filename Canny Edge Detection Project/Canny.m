%% Canny Edge Detector procedure
%1. Filter image with derivate of gaussian
%2. Find magnitude and orientation of gradient
%3. Non-maximum suppression
%4. Linking and thresholding(hysteresis):
%  -> define two thresholds: low and high
%  -> use the high threshold to start edge curves and the low thresohld to
%  continue them

% Key points! 1. Calculate Derivative of Gaussian in X and Y direction
%             2. Calculate the magnitude and direction using DoG(Derivative
%             of Gaussian)
%             3. Apply Non-Maximum suppression and Hysteresis thresholding

function imgResult = Canny(input_img, sigma, threshold, connectivity)
    % sigma = sigma value for the gaussian kernel
    % threshold = lower and upper value for hysteresis threshold (e.g. [100,
    % 200];)
    % connectivity = 4 or 8 connectivity for hysteresis threshold
    X = [-2,-1,0,1,2];
    Y = [-2;-1;0;1;2];
    % Get Gaussian Kernels in X and Y direction
    Gaussian_x = ((-X)/(sigma^2)).* exp(-((X.^2) / (2*sigma^2)));
    Gaussian_y = ((-Y)/(sigma^2)).* exp(-((Y.^2) / (2*sigma^2)));
    % Use Gaussian kernels to get derivative of gaussian in X and Y
    % direction (Convolution)
    DoG_X = conv2(input_img, Gaussian_x, 'same');
    DoG_Y = conv2(input_img, Gaussian_y, 'same');
    
    % Use Derivative of Gaussian in X and Y direction to get magnitude and
    % direction(in radians)
    Magnitude = sqrt(DoG_X.^2 + DoG_X.^2);
    Direction = (DoG_Y ./ DoG_X);
    
    % Non-maximum suppression
    % Bin the angles into 4 directions i.e. 0,45,90,135
    Degree = round(mod(radtodeg(Direction), 360));
    
    % Zero-padding the Magnitude
    [r,c] = size(Magnitude);
    Padded_Magnitude = zeros(r+2, c+2);
    Padded_Degree = zeros(r+2, c+2);
    
    [nr, nc] = size(Padded_Magnitude);
    Padded_Magnitude(2:nr-1, 2:nc-1) = Magnitude;
    Padded_Degree(2:nr-1, 2:nc-1) = Degree;
    
    % For every pixel, compare the gradient value with the neighbors along
    % the aforementioned directions and keep the maximum value of the
    % gradient image
    
    for i = 1:r
        for j =1:c
            % Get the center degree
            kernel_degree = Padded_Degree(i:i+2, j:j+2);
            center_degree = kernel_degree(2,2);
            
            % magnitude kernel
            kernel_mag = Padded_Magnitude(i:i+2, j:j+2);
            center_mag = kernel_mag(2,2);
            
            % find the degree which is same as the center degree
            coords = findCoordinates(center_degree, kernel_degree, connectivity);
            coords_size = size(coords,1);

            % find the magnitude's center and if center is less than
            % magnitude founded position(by degree)'s maximum value, set it
            % 0 
            
            % magnitude founded position(by degree)'s maximum value
            for a = 1: coords_size
                if kernel_mag(coords(a,1), coords(a,2)) > center_mag
                    kernel_mag(2,2) = 0;
                    break
                end
            end
            Padded_Magnitude(i:i+2, j:j+2) = kernel_mag;
        end
    end
    Magnitude = Padded_Magnitude(2: nr-1, 2:nc-1);
    
    % (d) Hysteresis Thresholding
    for i = 1:r
        for j = 1:c
            if Magnitude(i,j) >= min(threshold) & Magnitude(i,j) <= max(threshold)
                Magnitude(i,j) = 255;
            end
        end
    end
    imgResult = Magnitude;
end



