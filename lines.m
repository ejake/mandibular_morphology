function d = lines (r,n)
    %Points (coordinate x,y)
    %n = 5;
    %r = rand(n,2);
    d = zeros(1,factorial(n)/(factorial(2)*factorial(n-2)));
    l = 1;
    for i=1:n-1
        for j=i+1:n
            d(l) = ((r(i,1)-r(j,1))^2+(r(i,2)-r(j,2))^2)^.5;
            l = l+1;
        end
    end
    %disp(d);
end