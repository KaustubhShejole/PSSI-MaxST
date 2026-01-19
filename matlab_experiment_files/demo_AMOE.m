
clear all;close all;
addpath('.');
addpath('Ncut_9');
addpath('MatlabHPF');
addpath('imagesgrabcut');
addpath('algorithms');
addpath('onecut');
exampleDataDir = 'images250';

ims = dir([exampleDataDir '/images/*.*']);

outDir = fullfile(exampleDataDir, 'amoe_ssncut_scribbles');
markerDir = fullfile(exampleDataDir, 'ssncutmarker');


% Create output directory
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
for imgindex = 3:length(ims)
    
    imagename=ims(imgindex).name;
    imgFile = fullfile(exampleDataDir,['/images/' imagename]);
    img = imread(imgFile);
    img1=img;
    
     %% display original image
    figure; imshow(img); title('Original image');
       
    nei = 1;            % 0: 4-neighbors, 1: 8-neighbors
    scale = 1.0;        % image resize
    lambda = 2e-10;     % parameter for unitary
    imgName = 0; 
    
    ref_name = fullfile(markerDir, [ ims(imgindex).name(1:end-4) '_marker.bmp']);
    [K, labels, idx] = seed_generation(ref_name,scale);
    
    [X Y Z]=size(img); 
    N = X*Y; % image size
    
    boximage = fullfile(exampleDataDir,['/box/' ims(imgindex).name(1:end-4) '_box.bmp']);
    box=imread(boximage);
    boxone=box(:,:,1);
    boxsize=sum(boxone(:)==0); 
       
    
    %set energy parameters
	lambda = 9.0; % weight of smoothness term
	beta_prime = 0.9; % for L1 color separation term
    nei = 1;     % 0: 4-neighbors, 1: 8-neighbors
    channelstep=4;  %bin=(256/channelstep)^3
    INFTY=100000000;
    
    [points edges] = lattice(X,Y,nei); 
    imgVals = reshape(img,N,Z);
    imgVals=double(imgVals); 
    di=(imgVals(edges(:,1),1)-imgVals(edges(:,2),1)).^2+(imgVals(edges(:,1),2)-imgVals(edges(:,2),2)).^2+(imgVals(edges(:,1),3)-imgVals(edges(:,2),3)).^2;
	sigma_sum=sum(di(:,1));
    sigma_square_count=length(edges);
    sigma_square=sigma_sum/sigma_square_count;
    
    colorlabel=rgb2indeximg(img,channelstep);
    
    [colorbinnum, compacthist, colorlabelchange]= getcompactlabel(colorlabel);
    
    l1penalty=getl1penalty(colorlabelchange,box);
    
    beta = boxsize/l1penalty*beta_prime; % weight of L1 color separation term
    
    ROI=true(X,Y);   % Region of interest
    
    addsmoothnessterm_weight=addsmoothnessterm(edges, points, img, imgVals, lambda, ROI, sigma_square);
    
    [edges_addauxnode, weight_addauxnode]=addl1separationterm(edges, addsmoothnessterm_weight, colorlabelchange, beta, ROI);
    

     W = adjacency(edges_addauxnode,weight_addauxnode,N+colorbinnum); %clear edges weights;

    if (Z>1)
        I =double(rgb2gray(img));
    else
        I = double(img);
    end
    [w,imageEdges] = ICgraph(I);
     w(N+colorbinnum,N+colorbinnum)=0;
    
     W=W+w;
     if imgName == 0
        for k = 1:K
            seedsInd{k} =  idx(labels==k);
        end
     end
              
    source=seedsInd{1};
    sink=seedsInd{2};    
    
    sink2=find(boxone==255);
    unkown=find(boxone==0);
    
    Nsum=N+colorbinnum;
    W(Nsum+1,source)=INFTY;
    W(sink,Nsum+2)=INFTY;
    W(sink2,Nsum+2)=INFTY;
    W(Nsum+2,Nsum+2)=0;
    W=W-diag(diag(W));
   
    [value, cut]=hpf(W,Nsum+1,Nsum+2);
    cut1=cut(1:N);  
    
    seg=reshape(cut1,X,Y);
    figure;imagesc(seg);
    
    %在原图像上显示分割结果
    bw = edge(seg,0.01);
    [i,j] = find(bw(:,:));
    edgeindex=[i,j];
    fgColor = [255,0,0];
    imgseg=img1;

    [nr,nc]=size(edgeindex(:,1));
    for i = 1:3  
       for j=1:nr
           imgseg(edgeindex(j,1),edgeindex(j,2),i) = fgColor(i);   
       end 
    end
    figure;imshow(imgseg); %原图像上显示分割
    

    outputfile=fullfile(outDir, [ims(imgindex).name(1:end-4) '_hpfseg.bmp']);
    imwrite(imgseg,outputfile);
    outputfile=fullfile(outDir, [ims(imgindex).name(1:end-4) '.bmp']);
    imwrite(seg,outputfile);

    close all;
end
