
A=[F{1,1}(:,end:-1:1) , F{2,1}(:,:)];
A=cat(2,F{1,1}(:,end:-1:1) , F{2,1}(:,:));

B=cat(1,F{1,1}(:,end:-1:1) , F{3,1}(end:-1:1,:));



 A(1:2,1:2)=F{1,1}(:,end:-1:1)

A = 

    [24x24x24 double]    [24x24x24 double]    []    []
    [24x24x24 double]    [24x24x24 double]    []    []
                   []                   []    []    []
                   []                   []    []    []

A(1:2,3:4)=F{2,1}(:,:)

A = 

  Columns 1 through 3

    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]
    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]
                   []                   []                   []
                   []                   []                   []

  Column 4

    [24x24x24 double]
    [24x24x24 double]
                   []
                   []

A(3:4,1:2)=F{3,1}(end:-1:1,:)

A = 

  Columns 1 through 3

    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]
    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]
    [24x24x24 double]    [24x24x24 double]                   []
    [24x24x24 double]    [24x24x24 double]                   []

  Column 4

    [24x24x24 double]
    [24x24x24 double]
                   []
                   []

A

A = 

    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]
    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]    [24x24x24 double]
    [24x24x24 double]    [24x24x24 double]                   []                   []
    [24x24x24 double]    [24x24x24 double]                   []                   []

hold
Current plot held
Plot3DData(A{1,1},24,'*');
Plot3DData(A{1,2},24,'*');
Plot3DData(A{1,3},24,'*');
Plot3DData(A{1,4},24,'*');
Plot3DData(A{2,1},24,'*g');
Plot3DData(A{2,2},24,'*g');
Plot3DData(A{2,3},24,'*g');
Plot3DData(A{2,4},24,'*g');
Plot3DData(A{3,1},24,'*r');
Plot3DData(A{4,1},24,'*r');
Plot3DData(A{3,2},24,'*r');
Plot3DData(A{4,2},24,'*r');