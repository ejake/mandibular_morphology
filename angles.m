function ang = angles(r,n)
    %Points (coordinate x,y)
    %n = 5;
    %r = rand(n,2);
    %ang = zeros(factorial(n)/(factorial(3)*factorial(n-3)),3);
    ang = zeros(1,(factorial(n)/(factorial(3)*factorial(n-3)))*3);
    l = 1;
    for i=1:n-1
        for j=i+1:n
            for k=j+1:n
                s1 = ((r(i,1)-r(j,1))^2+(r(i,2)-r(j,2))^2)^0.5;
                s2 = ((r(j,1)-r(k,1))^2+(r(j,2)-r(k,2))^2)^0.5;
                s3 = ((r(k,1)-r(i,1))^2+(r(k,2)-r(i,2))^2)^0.5;
                ang(l) = acos((s1^2+s3^2-s2^2)/(2*s1*s3));%i			
                ang(l+1) = acos((s2^2+s1^2-s3^2)/(2*s2*s1));%j			
                ang(l+2) = acos((s3^2+s2^2-s1^2)/(2*s3*s2));%k
                l = l+3;
            end
        end
    end
    %disp(ang);
end
