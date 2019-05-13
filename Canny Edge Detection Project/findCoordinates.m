function coord = findCoordinates(center, kernel, connectivity)
    coord = [];
    center = floor(center/45) * 45;
    for i = 1:3
        for j = 1:3
            kernel(i,j) = floor(kernel(i,j) /45) * 45;
        end
    end
    
    
    if connectivity == 4
        if kernel(1,2) == center
            coord = [coord; [1,2]];
        end
        if kernel(2,1) == center
            coord = [coord; [2,1]];
        end
        if kernel(2,3) == center
            coord = [coord; [2,3]];
        end
        if kernel(3,2) == center
            coord = [coord; [3,2]];
        end             
    end
    
    if connectivity == 8
        for i=1:3
            for j=1:3
                if i ~= 2 & j ~= 2 
                    if kernel(i,j) == center
                        coord = [coord; [i,j]];
                    end
                end
            end
        end
    end
end