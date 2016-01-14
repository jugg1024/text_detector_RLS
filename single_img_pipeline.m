% Author : Gen.Li
% Paper :  Scene text detection with Extremal region based Cascaded Filtering     
% Department: Insititute of automation, chinese academy of science
% Email : jugg1024@gmail.com

%% code : this is the code of my work; pipeline of a single image, Lite Version;
%% 0. Preparation
%% 0.0 pre-load, include svm models,CNN models 
% SVM model for broad and narrow type
tic
narrow_model = load('3rd_party_tools/narrow_svmStruct_Dec_06');
narrow_pca = load('3rd_party_tools/narrow_pca_data_Dec_06');
broad_model = load('3rd_party_tools/broad_svmStruct_Dec_06');
broad_pca = load('3rd_party_tools/broad_pca_data_Dec_06');
G=fspecial('gaussian',3,2);
Cato_Names = {'0' 'l' '2' '3' '4' '5' '6' '7' '8' '9'...
               ')' '&' '(' ' ' ' ' ' ' ' ' 'A' 'B' 'C'...
               'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M'...
               'N' ' ' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W'...
               'X' 'Y' ' ' ' ' ' ' ' ' ' ' ' ' ' ' 'a'... 
               'b' ' ' 'd' 'e' 'f' 'g' 'h' 'i' 'j' ' '...
               ' ' 'm' 'n' ' ' ' ' 'q' 'r' ' ' 't' 'u'...
               ' ' ' ' ' ' 'y' ' ' '?'...
               };
max_size = 1000;
B = [0.947 0.295 -0.131;-0.118 0.993 0.00737;0.0923 -0.0465 0.995];
A = [27.1 -22.8 -1.81;-5.65 -7.72 12.9;-4.16 -4.58 -4.58];
resize = [26,26;26,10];
border = [4,29,4,29;4,29,12,21];
%% 0.1 data input
img_name = 'img_1.jpg';
rgb = imread(img_name);   
rgb_cp = rgb;
[m,n,~] = size(rgb);
% resize to speed up
scale = 1;
if max(m,n) > max_size
    scale = max_size/max(m,n);
    rgb = imresize(rgb,scale,'bilinear');
end    
[m,n,p] = size(rgb);
%% 1. Character Candidate Extraction in Multi-channel 
%% 1.1 Multi-channel image transform
ycbcr = rgb2ycbcr(rgb);
rgb = double(rgb)/255;
xyz = rgb2xyz(rgb);
x = xyz(:,:,1);y = xyz(:,:,2);z= xyz(:,:,3);
xyz_sensor = cat(2,x(:),y(:),z(:));
pii_xyz_sensor = A * log(B * xyz_sensor'); pii_xyz_sensor =  pii_xyz_sensor';    
pii_xyz = reshape(pii_xyz_sensor, [m n p]);
pii_rgb = xyz2rgb(pii_xyz);
pii_rgb = real(pii_rgb);
pii_rgb = uint8(floor(pii_rgb * 255));
pii_hsv = rgb2hsv(pii_rgb);
clear xyz xyz_sensor pii_xyz_sensor pii_xyz pii_rgb;
% eight intensity maps; gray in RGB, hue in piiHSV, sat in piiHSV ,cb in YCBCR; and their inverted maps    
gray = rgb2gray(rgb);
red = rgb(:,:,1) ;g = rgb(:,:,2) ;b = rgb(:,:,3) ;
pii_hue = pii_hsv(:,:,1);
pii_sat = pii_hsv(:,:,2); 
cb = double(ycbcr(:,:,2))/255;
intensity_maps = cat(3,gray,1-gray,cb,1-cb,pii_hue,1-pii_hue,pii_sat,1-pii_sat); 
rgb_sensor = cat(2,red(:),g(:),b(:)) * 255;
%% 1.2 Extract ER candidate from this eight color maps using vl-mser.            
TLC = [];
for channel = 1:8
    I = intensity_maps(:,:,channel);
    I = uint8(floor(I * 255));
    % first extract ER with: delta 3; max variation 0.5 ; mindiversity 0.1
    [r,f] = vl_mser(I,'MinDiversity',0.1,...
                'MaxVariation',0.5,...
                'MaxArea',0.5,...
                'MinArea',22.3607 / (m * n),...
                'BrightOnDark',1,...
                'DarkOnBright',0,...
                'Delta',3) ;
    f = vl_ertr(f) ;
    er = [];
    % get the information of ER
    for i = 1:length(r)
        s = vl_erfill(I,r(i));
        left = min(floor((double(s)-1)/double(m)) + 1);
        right = max(floor((double(s)-1)/double(m)) + 1);
        up = double(min(mod(s-1,m) + 1));
        down = double(max(mod(s-1,m) + 1));
        er(i).width = double(right - left + 1);
        er(i).height = double(down - up + 1);
        er(i).rect = [up down left right];
        er(i).area = length(s);
        er(i).fill_ratio = er(i).area / (er(i).width * er(i).height);
        er(i).aspect_ratio = er(i).width / er(i).height;
        er(i).total_frames = s;
    end      
    %% 2. Character Candidate Classification
    %% 2.1 Geometry pre-pruning 
    if ~isempty(er)
        avr_width = mean([er.width]);
        avr_height = mean([er.height]);
        avr_area = mean([er.area]);
    end
    narrow_f = []; narrow_idx = [];  narrow_rect = []; 
    broad_f = []; broad_idx = []; broad_rect = []; 
    for i = 1:length(r)            
        % remove too small or too large or too full or too empty or too narrow or too fat   
        if (er(i).width > 0.85*n && avr_width < 0.5*n) || (er(i).height>0.85*m && avr_height<0.4*m) || (er(i).area>0.3*m*n && avr_area <0.1*m*n) ||... % too large
            er(i).width< 7 ||  er(i).height< 7 || (er(i).width < 9 && er(i).height < 9) || er(i).area < 40 ||...       % too small
            (er(i).area < 100 && er(i).fill_ratio > 0.7) || (er(i).aspect_ratio > 0.35 && er(i).fill_ratio > 0.8) || er(i).fill_ratio < 0.201 || ... % too full or too empty                                                       % too full or empty
            er(i).aspect_ratio < 0.15 || er(i).aspect_ratio > 2.3 % too narrow or too fat   
            % removed
        else
            if er(i).aspect_ratio<=0.4
                resz = 2;
            elseif er(i).aspect_ratio <=1.75
                resz = 1;
            else
                resz = 0;
            end
            if resz>0
                er(i).color = mean(rgb_sensor(er(i).total_frames,:),1);
                I_binary = ones(size(I))*255;
                I_binary(er(i).total_frames) = 0;
                rect = er(i).rect;
                I_binary = I_binary(rect(1):rect(2),rect(3):rect(4));
                img = ones(32,32) * 255;
                I_binary = imresize(I_binary,resize(resz,:));
                img(border(resz,1):border(resz,2),border(resz,3):border(resz,4)) = I_binary;
                img = imfilter(img,G,'replicate');  img = uint8(img);
                f = extractHOGFeatures(img,'CellSize',[4 4],'BlockSize',[3 3],'BlockOverlap',[1 1],'UseSignedOrientation',true);
                f2 = lbp(img); f2 = f2 /sum(f2);
                f = [f f2];
                if resz == 2 % narrow    
                    narrow_f = [narrow_f f'];
                    narrow_idx = [narrow_idx i];
                    narrow_rect = [narrow_rect rect'];    
                elseif resz == 1 % broad  
                    broad_f = [broad_f f'];
                    broad_idx = [broad_idx i];
                    broad_rect = [broad_rect rect'];
                end
            end
        end
    end
    
     
    position1 = [];position2 = [];position3 = [];
    RGB = I;
    if ~isempty(narrow_rect)
        position1(1,:) = narrow_rect(3,:);
        position1(2,:) = narrow_rect(1,:);
        position1(3,:) = narrow_rect(4,:) - narrow_rect(3,:) + 1;
        position1(4,:) = narrow_rect(2,:) - narrow_rect(1,:) + 1;
        RGB = insertObjectAnnotation(RGB, 'rectangle', position1', 'Ner','Color', 'cyan', 'TextColor', 'black');
    end
    if ~isempty(broad_rect)
        position2(1,:) = broad_rect(3,:);
        position2(2,:) = broad_rect(1,:);
        position2(3,:) = broad_rect(4,:) - broad_rect(3,:) + 1;
        position2(4,:) = broad_rect(2,:) - broad_rect(1,:) + 1;
        RGB = insertObjectAnnotation(RGB, 'rectangle', position2', 'Ber','Color', 'yellow', 'TextColor', 'black');
    end   
    
    %% 2.2 Classify the candidate using the ML method
    if ~isempty(narrow_f)
        narrow_f = (double(narrow_f') - repmat(narrow_pca.mean_vec,size(narrow_f,2),1)) * narrow_pca.COEFF; narrow_f = narrow_f(:,1:narrow_pca.pos);
    end
    if ~isempty(broad_f)
        broad_f = (double(broad_f') - repmat(broad_pca.mean_vec,size(broad_f,2),1)) * broad_pca.COEFF; broad_f = broad_f(:,1:broad_pca.pos);
    end 
    narrow_predict_label = libsvmpredict(narrow_idx',double(narrow_f),narrow_model.model, '-b 1');
    pos = find(narrow_predict_label<75); 
    broad_predict_label = libsvmpredict(broad_idx',double(broad_f),broad_model.model, '-b 1');
    pos2 = find(broad_predict_label<75); 
    

    RGB2 = I;
    if ~isempty(pos)
        g_narrow_predict_label = narrow_predict_label(pos);
        position1 = position1(:,pos);
        RGB2 = insertObjectAnnotation(RGB2, 'rectangle', position1', Cato_Names(g_narrow_predict_label+1),'Color', 'cyan', 'TextColor', 'black');
    end  
    if ~isempty(pos2)
        g_broad_predict_label = broad_predict_label(pos2);
        position2 = position2(:,pos2);
        RGB2 = insertObjectAnnotation(RGB2, 'rectangle', position2', Cato_Names(g_broad_predict_label+1),'Color', 'yellow', 'TextColor', 'black');
    end
        
    %% 3. Character properties based Local search
    removed_id = [];
    % onlly broad and some narrow numbers(0,2,9) are used for seed.
    if channel <= 2 || channel >= 7
        search_seed_idx = [broad_idx(pos2) narrow_idx(narrow_predict_label==0|narrow_predict_label==2|narrow_predict_label==9)];
        removed = [narrow_idx(narrow_predict_label==75) broad_idx(broad_predict_label==75)];
        rer = er(removed);
        if ~isempty(rer)
            colors = reshape([rer.color]',[3 length(rer)])'; 
            rects = reshape([rer.rect]',[4 length(rer)])';  
            heights = [rer.height]';
            widths = [rer.width]';
            aspects = widths./heights;
            fillrates = [rer.fill_ratio]';
            x_centers = (rects(:,3) + rects(:,4))/2 ;
            y_centers = (rects(:,1) + rects(:,2))/2;
            areas = heights .* widths;
        end
        find_back = zeros(length(rer),1);
        if ~isempty(search_seed_idx) && ~isempty(removed)
            seed_rect = [broad_rect(:,pos2)' ;narrow_rect(:,narrow_predict_label==0|narrow_predict_label==2|narrow_predict_label==9)'];
            seed_height = seed_rect(:,2) - seed_rect(:,1) + 1;
            seed_width = seed_rect(:,4) - seed_rect(:,3) + 1;
            for i = 1:length(search_seed_idx)
                lid = search_seed_idx(i);
                rid = search_seed_idx(i);
                while lid
                    l_rect = er(lid).rect;
                    l_y_center = (l_rect(1) + l_rect(2))/2; l_x_center = (l_rect(3) + l_rect(4))/2;
                    l_h = l_rect(2) - l_rect(1) + 1;         l_w = l_rect(4) - l_rect(3) + 1;
                    color_dif = colors - repmat(er(lid).color,length(rer),1);
                    color_thresh = 20 * (1 -  max(l_h, l_w)/min(m,n));
                    left = find( sqrt(sum(color_dif.^2,2)) < color_thresh & abs( y_centers -  l_y_center) < 0.45 * l_h & abs(x_centers - l_x_center) < 3 * max(l_w,l_h) ...
                                &  abs( heights - l_h ) < 0.4 * l_h & abs( widths - l_w ) < 5 * l_w &...
                                ~(fillrates - er(lid).fill_ratio > 0.3 & aspects > 0.4) & ( x_centers < l_rect(3) ) );
                    true_left_idx = [];
                    for le = 1:length(left)
                        rec = rects(left(le),:);
                        rec_h = rec(2) - rec(1) + 1;
                        dif = seed_rect - repmat(rec,size(seed_rect,1),1);
                        if find((dif(:,1)<4 & dif(:,2)>-4 & dif(:,3)<4 & dif(:,4)>-4) & rec_h > 0.4 * seed_height & seed_height < 0.25 * m & seed_width < 0.25 * n)
                        elseif find((dif(:,1)>-4 & dif(:,2)<4 & dif(:,3)>-4 & dif(:,4)<4) & rec_h < 2 * seed_height)
                        elseif find_back(left(le)) == 1;
                        else
                            find_back(left(le)) = 1;
                            true_left_idx = [true_left_idx left(le)];
                        end
                    end
                    true_left = rects(true_left_idx,:);
    %                         [~,ind] = min(true_left(:,3)+true_left(:,4));
                    h_dif = abs(true_left(:,2) - true_left(:,1) + 1 - l_h);
                    [~,ind] = min(h_dif);
                    lid = removed(true_left_idx(ind));
                end
                while rid
                    r_rect = er(rid).rect;
                    r_y_center = (r_rect(1) + r_rect(2))/2; r_x_center = (r_rect(3) + r_rect(4))/2;
                    r_h = r_rect(2) - r_rect(1) + 1;         r_w = r_rect(4) - r_rect(3) + 1;
                    color_dif = colors - repmat(er(rid).color,length(rer),1);
                    color_thresh = 20 * (1 - max(r_h, r_w)/min(m,n));
                    right = find( sqrt(sum(color_dif.^2,2)) < color_thresh  & abs( y_centers -  r_y_center) < 0.45 * r_h & abs(x_centers - r_x_center) < 3 * max(r_w,r_h) ...
                                &  abs( heights - r_h ) < 0.4 * r_h & abs( widths - r_w ) < 5 * r_w &...
                                ~(fillrates - er(rid).fill_ratio > 0.3 & aspects > 0.4) & (  x_centers > r_rect(4) ) );
                    true_right_idx = [];
                    for ri = 1:length(right)
                        rec = rects(right(ri),:);
                        rec_h = rec(2) - rec(1) + 1;
                        dif = seed_rect - repmat(rec,size(seed_rect,1),1);
                        if find((dif(:,1)<4 & dif(:,2)>-4 & dif(:,3)<4 & dif(:,4)>-4) & rec_h > 0.4 * seed_height & seed_height < 0.25 * m & seed_width < 0.25 * n)
                        elseif find((dif(:,1)>-4 & dif(:,2)<4 & dif(:,3)>-4 & dif(:,4)<4) & rec_h < 2 * seed_height)
                        elseif find_back(right(ri)) == 1;
                        else
                            find_back(right(ri)) = 1;
                            true_right_idx = [true_right_idx right(ri)];
                        end
                    end
                    true_right = rects(true_right_idx,:);
    %                         [~,ind] = max(true_right(:,3)+true_right(:,4));
                    h_dif = abs(true_right(:,2) - true_right(:,1) + 1 - r_h);
                    [~,ind] = min(h_dif);
                    rid = removed(true_right_idx(ind));
                end
            end
        end
        removed_id = removed(find_back == 1);    
    end
    tlc_id = [narrow_idx(pos) broad_idx(pos2)];
    tlc_id = [tlc_id removed_id];     
    
    RGB3 = RGB2;
    if ~isempty(removed_id)
        findba = rer(find_back==1);
        find_back_rects = reshape([findba.rect]',[4 length(findba)]);  
        position3(1,:) = find_back_rects(3,:);
        position3(2,:) = find_back_rects(1,:);
        position3(3,:) = find_back_rects(4,:) - find_back_rects(3,:) + 1;
        position3(4,:) = find_back_rects(2,:) - find_back_rects(1,:) + 1;
        find_back_label = 75 * ones(size(findba,1),1);
        RGB3 = insertObjectAnnotation(RGB3, 'rectangle', position3', Cato_Names(find_back_label+1),'Color', 'red', 'TextColor', 'black');
    end

    for i = 1:length(tlc_id)
        er_cc = er(tlc_id(i));
        tlc_cc.color = er_cc.color ; 
        tlc_cc.var = norm(std(rgb_sensor(er_cc.total_frames,:),0,1)) ;
        tlc_cc.width = er_cc.width;
        tlc_cc.height = er_cc.height;
        tlc_cc.area = er_cc.area;
        tlc_cc.fill_ratio = er_cc.fill_ratio;
        tlc_cc.rect = er_cc.rect;
        tlc_cc.channel = channel;
%         tlc_cc.totalframes = er_cc.total_frames;
        I_binary = ones(size(I))*255;
        I_binary(er_cc.total_frames) = 0;
        rect = er_cc.rect;
        I_binary = I_binary(rect(1):rect(2),rect(3):rect(4));
        edges = edge(I_binary,'canny');
        L = sum(sum(edges>0));
        tlc_cc.sw = 2*er_cc.area/L;
        if i <= length(pos)
            tlc_cc.label = narrow_predict_label(pos(i));
        elseif i <= length(pos) + length(pos2)
            tlc_cc.label = broad_predict_label(pos2(i - length(pos)));
        else
            tlc_cc.label = 75;
        end
        TLC = [TLC tlc_cc];
    end  
end


if ~isempty(TLC);
    tlc_re = reshape([TLC.rect]',[4 length(TLC)])';    
    tlc_la = [TLC.label]';
    draw_rec = [];
    draw_rec(:,1) = tlc_re(:,3); 
    draw_rec(:,2) = tlc_re(:,1);
    draw_rec(:,3) = tlc_re(:,4) - tlc_re(:,3)+1;
    draw_rec(:,4) = tlc_re(:,2) - tlc_re(:,1)+1;
    RGB1 = insertObjectAnnotation(rgb, 'rectangle', draw_rec, Cato_Names(tlc_la+1) ,'Color', 'yellow', 'TextColor', 'black');   
end


%% 4. Duplicate removal
%% 4.1 Remove multi-channel duplicates
if ~isempty(TLC)
    duplica = zeros(length(TLC),1);
    rects = reshape([TLC.rect]',[4 length(TLC)])';
    for tlc = 1:length(TLC)
        if duplica(tlc) == 0
            dif = rects - repmat(rects(tlc,:),length(TLC),1);
            absdif = abs(dif);
            rec = TLC(tlc).rect;
            thres = max( 5, 3 + 0.03 * max(rec(2)-rec(1)+1 , rec(4)-rec(3)+1) );
            pos = find(absdif(:,1)<thres & absdif(:,2)<thres & absdif(:,3)<thres & absdif(:,4)<thres );
            pos2 = pos([TLC(pos).label]<75); 
            if isempty(pos2)
                pos2 = pos;
            end
            vars = [TLC(pos2).var];
            [~,ind] = min(vars);
            duplica(pos(pos ~= pos2(ind))) = 1;  
        end
    end
    TLC(duplica == 1) = [];
    rects = reshape([TLC.rect]',[4 length(TLC)])';
end
%% 4.2 Remove RLS duplicates     
duplica = zeros(length(TLC),1);
TLC_vars = [TLC.var]';
ave_var = mean(TLC_vars);
TLC_labels = [TLC.label]';
TLC_heights = [TLC.height]';
TLC_widths = [TLC.width]';
for tlc = 1:length(TLC)
    if tlc == 27
        limg=1;
    end
    if (TLC(tlc).width / TLC(tlc).height < 0.28 && TLC(tlc).height > 0.5 * m && TLC(tlc).height < 0.78 * m) || ... % remove too narrow & high     
       (TLC(tlc).width / TLC(tlc).height < 0.22 && TLC(tlc).fill_ratio < 0.25 ) || ... % remove too narrow & empty       
       (TLC(tlc).label == 1 && TLC(tlc).channel == 7 &&TLC(tlc).sw <4 &&TLC(tlc).width <17 &&TLC(tlc).fill_ratio <0.33 )||...
       (TLC(tlc).label == 26 && TLC(tlc).channel == 8 &&TLC(tlc).sw <5.7 &&TLC(tlc).height <32 &&TLC(tlc).fill_ratio <0.26 )||...
       (TLC(tlc).label == 75 && TLC(tlc).channel == 7 &&TLC(tlc).sw <4.67 &&TLC(tlc).width/TLC(tlc).height > 1.4 &&TLC(tlc).fill_ratio <0.27 )||...
       (TLC(tlc).label == 75 && ( TLC(tlc).var > max(80 , 3 * ave_var) || (TLC(tlc).fill_ratio < 0.205 && TLC(tlc).channel < 3)...
        || (TLC(tlc).fill_ratio < 0.2308 && TLC(tlc).channel > 6) ) ) ||... % remove high var, empty of those added
       ( (TLC(tlc).label == 0 || TLC(tlc).label == 33)&&(TLC(tlc).sw < 2.7) +(TLC(tlc).fill_ratio < 0.223) + (TLC(tlc).var > 70)>=2) % remove bad 0&o
        duplica(tlc) = 1;
    end
     % remove duplicate added
     % when covered area recall& precision both upon 0.5 
    rep_tlc = repmat(TLC(tlc).rect,length(TLC),1);
    dif = rects - rep_tlc;  absdif = abs(dif);
    max_l = max(rects(:,3),rep_tlc(:,3));
    min_r = min(rects(:,4),rep_tlc(:,4));
    max_u = max(rects(:,1),rep_tlc(:,1));
    min_d = min(rects(:,2),rep_tlc(:,2));
    dup_size = max( (min_r - max_l + 1) , 0 ) .*  max( (min_d - max_u + 1) , 0 );
    fin = repmat(TLC(tlc).width*TLC(tlc).height,length(TLC),1);
    tru = TLC_heights .* TLC_widths;
    recall = dup_size./tru; precision = dup_size./fin;
    if  TLC(tlc).label ==75 && ... 
        ( sum( dif(:,1)<4 & dif(:,2)>-4 & dif(:,3)<4 & dif(:,4)>-4 & TLC_labels<75 & ...
        TLC_heights < 0.25 * m & TLC_widths < 0.25 * n  ) > 0 || ...
        sum( TLC_labels<75 & recall > 0.5 & precision > 0.49 ) > 0)
        duplica(tlc) = 1;
    end
end    
%% 4.3 Remove out-of-ranged and partly-covered duplicates 
TLC(duplica == 1) = [];
rects(duplica == 1,:) = [];
% select the most suitable rect , remove other duplicate
duplica = zeros(length(TLC),1);
TLC_vars = [TLC.var]';
ave_var = mean(TLC_vars);
TLC_labels = [TLC.label]';
TLC_heights = [TLC.height]';
TLC_widths = [TLC.width]';
for tlc = 1:length(TLC)
    if tlc == 30
        limg=1;
    end
    if duplica(tlc) == 0
        % remove ouside rect(bad propoties)
        dif = rects - repmat(TLC(tlc).rect,length(TLC),1);  absdif = abs(dif);
        if sum((dif(:,1)<6 & dif(:,2)>-6 & dif(:,3)<6 & dif(:,4)>-6)) > 1
            % remove too fill_less with high color variation(outside)
            pos = find(dif(:,1)<6 & dif(:,2)>-6 & dif(:,3)<6 & dif(:,4)>-6 & TLC_heights < 4*TLC_heights(tlc));
            pos = pos(pos~=tlc);
            vars = [TLC(pos).var];
            fillrates = [TLC(pos).fill_ratio];
            duplica(pos(vars > max(50,3.134*ave_var) | (fillrates < 0.251 & vars > 2*ave_var)) ) = 1;

            % remove duplicate, select the biggest one in those with low variation 
            pos = find(dif(:,1)<6 & dif(:,2)>-6 & dif(:,3)<6 & dif(:,4)>-5 &  ...
                 ( (absdif(:,1)<20 & absdif(:,2)<20) | (absdif(:,3)<20 & absdif(:,4)<20) |...
                 ( TLC_heights < 1.6 *TLC(tlc).height & TLC_widths < 1.6 *TLC(tlc).width  ) ) & duplica == 0);
            if ~isempty(pos) && TLC(tlc).area >= 110
                % low variation candidate
                fillrates = [TLC(pos).fill_ratio]; mean_fill = mean(fillrates);
                cand = pos( [TLC(pos).var] < min(2.05*ave_var,80) & [TLC(pos).fill_ratio] > min(0.75 * mean_fill,0.235));   
                % biggest size
                if ~isempty(cand)
                    [~,ind] = max([TLC(cand).area]);
                    duplica(pos(pos ~= cand(ind))) = 1; 
                else
                    [~,ind] = max([TLC(pos).area]);
                    duplica(pos(pos ~= pos(ind))) = 1; 
                end
            end
        end
    end
end
%% 4.4 Remove extreme small(less than 60) inside candidates 
TLC(duplica == 1) = [];
rects(duplica == 1,:) = [];
% remove rect inside small rect(less than 60)
duplica = zeros(length(TLC),1);
TLC_vars = [TLC.var]';
ave_var = mean(TLC_vars);
TLC_labels = [TLC.label]';
TLC_heights = [TLC.height]';
TLC_widths = [TLC.width]';
for tlc = 1:length(TLC)
    if tlc == 58
        limg=1;
    end
    if duplica(tlc) == 0
        rep_tlc = repmat(TLC(tlc).rect,length(TLC),1);
        dif = rects - rep_tlc;  absdif = abs(dif);
        if sum(dif(:,1)>-4 & dif(:,2)<4 & dif(:,3)>-4 & dif(:,4)<4) > 1 && min(TLC(tlc).width,TLC(tlc).height) < 60
           pos = find(dif(:,1)>-4 & dif(:,2)<4 & dif(:,3)>-4 & dif(:,4)<4);
           pos = pos(pos~=tlc);
           duplica(pos) = 1; 
        end
        % remove rect ouside numbers 
        if  TLC_labels(tlc) < 10 && TLC_labels(tlc) >=2 
            pos = find((dif(:,1)<1 & dif(:,2)>-1 & dif(:,3)<1 & dif(:,4)>-1) & TLC_heights < 2 * TLC_heights(tlc));
            pos = pos(pos~=tlc);
            duplica(pos) = 1;
        end
    end
end
TLC(duplica == 1) = [];
rects(duplica == 1,:) = [];
[~,ind] = sort(rects(:,3));
TLC = TLC(ind);
rects = rects(ind,:);

position = [];
position(:,1) = rects(:,3);
position(:,2) = rects(:,1);
position(:,3) = rects(:,4) - rects(:,3) + 1;
position(:,4) = rects(:,2) - rects(:,1) + 1;
RGB2 = insertObjectAnnotation(rgb, 'rectangle', position, 1:length(TLC),'Color', 'yellow', 'TextColor', 'black');


%% 5. Character Grouping 
%% 5.1 Grouping using character Properties    
Line = [];
for i = 1:length(TLC)
    cur = TLC(i);
    if i == 29
        limg=16;
    end
    merge = 0;
    for j = 1:length(Line)
        right = TLC(Line(j).right_idx);
        % height
        Rh = min(cur.height,right.height) / max(cur.height,right.height);
        Lv = min(cur.rect(2),right.rect(2)) - max(cur.rect(1),right.rect(1)) +1;
        Rv = Lv / max(cur.height,right.height);
        % color
        colors = reshape([TLC(Line(j).set).color]',[3 length(Line(j).set)])';  
        Dc = min( sqrt(sum( (colors - repmat(cur.color , length(Line(j).set),1)).^2,2))/1.732);
        % distance of diffrent situation
        Dx = abs(cur.rect(3) - right.rect(4)) / (Line(j).ave_gap+ 20/(scale/3 + 0.6666));
        if Rv < 0.4 && cur.rect(3) - right.rect(4) <0
            Dx = 2 * abs(cur.rect(3) - right.rect(4)) / (Line(j).ave_gap);
        end
        % theta
        THETAt = (atan( (cur.rect(1)-right.rect(1))/(cur.rect(3) - right.rect(4) + cur.width/2 + right.width/2) ) )/pi*180;
        THETAb = (atan( (cur.rect(2)-right.rect(2))/(cur.rect(3) - right.rect(4) + cur.width/2 + right.width/2) ) )/pi*180;
        % y_dif
        difA = abs(cur.rect(1)-right.rect(1));
        difB = abs(cur.rect(2)-right.rect(2));
        y_dif = min(difA,difB);
        % sw_dif
        if cur.sw == Inf || right.sw == Inf || cur.sw > cur.width || right.sw > right.width
            sw_ratio = 1;
        else
            sw_ratio = cur.sw/right.sw;
        end

        if(Line(j).len == 1)
            if abs(THETAt)<abs(THETAb)
                theta = THETAt;
            else
                theta = THETAb;
            end
            delta = abs(theta);
        else
            deltat = abs(Line(j).last_theta - THETAt);
            deltab = abs(Line(j).last_theta - THETAb);
            delta = min(deltat,deltab);
            if deltat<deltab
                theta = THETAt;
            else
                theta = THETAb;
            end
        end
        if Line(j).ave_height / max(m,n) > 0.02
            Tdx = 3;
        else
            Tdx = 3.4;
        end
%             Trh = min(0.125*Dx+0.359,0.7);
        Trh = min(0.128*Dx+0.35,0.7);
        Trv = min(0.11*Dx+0.32,0.7);
        Tdc = 62 - 17.5*Dx;
%             Tdc2 = 100;
        Del = max(15,20-5*Dx);
        Dydif = max(10, (0.5-Dx) * max(cur.height, right.height));
        if (cur.label == 75 && right.label <10) || (cur.label == 75 && right.label <10)
            score = (Dc - 20)*1.5 + ( 0.8 - Rv ) * 100 + (max(sw_ratio,1/sw_ratio) - 1.5) * 10;
            Tscore = 40;
        elseif cur.height < 0.44 * right.height && cur.width < 0.27 * right.width
            score = (Dc - 25) + ( 0.7 - Rv ) * 160 + (max(sw_ratio,1/sw_ratio) - 1.5) * 10;
            Tscore = 30;
        else
            score = (Dc - 25) + ( 0.7 - Rv ) * 80 + (max(sw_ratio,1/sw_ratio) - 1.5) * 10;
            Tscore = 50;
        end 
        if Dx < Tdx && Rh > Trh && Rv > Trv && Dc < Tdc && score<Tscore &&(delta < Del || (Dx<0.5 && Dc < 30 && y_dif<Dydif  && sw_ratio >=0.3 && sw_ratio<=3) )
            if cur.rect(4) - right.rect(4) < 0 
                if cur.height< 0.5 * right.height && cur.width< 0.5 * right.width
                    % new Line
                else
                    %  this line but not update the line propties
                    merge = 1;
                    break;
                end
            else
                Line(j).right_idx = i;
                Line(j).len = Line(j).len + 1;
                Line(j).last_theta = theta;
                if ~(cur.width/cur.height<0.3 && Line(j).ave_width / Line(j).ave_height>0.5 )
                    Line(j).ave_width = (length(Line(j).set) * Line(j).ave_width + cur.width)/(length(Line(j).set) + 1);
                end
                Line(j).ave_height = (length(Line(j).set) * Line(j).ave_height + cur.height)/(length(Line(j).set) + 1);
                if cur.rect(3) - right.rect(4) + 1 > 0
                    Line(j).ave_gap = (length(Line(j).set) * Line(j).ave_gap + cur.rect(3) - right.rect(4) + 1)/(length(Line(j).set) + 1);              
                end
                Line(j).set = [Line(j).set i];
                merge = 1;
                break;
            end    
        end
    end
    if merge ==0
        line.right_idx = i;
        line.len = 1;
        line.last_theta = 0;
        line.ave_width = cur.width;
        line.ave_height = cur.height;
        line.ave_gap = (cur.height+cur.width)/2;
        line.set = i;
        Line = [Line line];
    end
end
%% 5.2 textLine refinement
% remove Line that has one word and too narrow or too small
Line([Line.len] == 1 & [TLC([Line.right_idx]).label] ~= 35  & ( ( max([Line.ave_width],[Line.ave_height]) < 0.021 * max(m,n))  | ( [Line.ave_width]./[Line.ave_height]< 0.46) ) ) = [];

% remove too different 'letter' in Line (broad is too large than average,var is too large than average)
for mm = 1:length(Line)
    vars = [TLC(Line(mm).set).var]; 
    labels = [TLC(Line(mm).set).label]; 
    broads = [TLC(Line(mm).set).width] .* [TLC(Line(mm).set).height];
    mean_var = mean(vars); mean_broad = mean(broads);
    Line(mm).set(vars > max(2.9 * mean_var ,55) | broads > 3.576 * mean_broad ) = [];
    while length(Line(mm).set) >=3 && TLC(Line(mm).set(1)).label == 75
        vars = [TLC(Line(mm).set).var]; 
        labels = [TLC(Line(mm).set).label]; 
        broads = [TLC(Line(mm).set).width] .* [TLC(Line(mm).set).height];
        mean_var = mean(vars); mean_broad = mean(broads);
        score1 =  max ( 2 * (vars(1) / mean_var -1) ,0 );
        score2 =  max ( (0.25 - TLC(Line(mm).set(1)).fill_ratio ) * 100 , 0);
        ave_gap = (Line(mm).ave_gap * length(Line(mm).set) - (TLC(Line(mm).set(1)).width+TLC(Line(mm).set(1)).height)/2 ...
            - (TLC(Line(mm).set(2)).rect(3) - TLC(Line(mm).set(1)).rect(4)) + 1) / (length(Line(mm).set) - 2);
        score3 = max( (TLC(Line(mm).set(2)).rect(3) - TLC(Line(mm).set(1)).rect(4) + 1) /ave_gap  - 1 , 0 ) * 5 * (length(Line(mm).set) - 2)^0.2;
        if  score1 + score2 + score3 > 10
            Line(mm).set(1) = [];
            Line(mm).len = Line(mm).len - 1;
        else
            break;
        end
    end
    if length(Line(mm).set) == 2
        if TLC(Line(mm).set(1)).label < 10 && TLC(Line(mm).set(1)).label > 1 && TLC(Line(mm).set(2)).label == 75
            number = Line(mm).set(1);
            bad = Line(mm).set(2);
            bad_in = 2;
        elseif TLC(Line(mm).set(1)).label == 75 && TLC(Line(mm).set(2)).label < 10 && TLC(Line(mm).set(2)).label > 1
            number = Line(mm).set(2);
            bad = Line(mm).set(1);
            bad_in = 1;
        else
            bad = 0;
        end

        if bad > 0 && (TLC(bad).channel == 2 ||TLC(bad).channel == 8) && ( TLC(bad).fill_ratio <0.21 ||...
                (TLC(bad).width/TLC(bad).height > 1.48 &&TLC(bad).fill_ratio <0.257 ) ||...
                (TLC(Line(mm).set(2)).rect(3) - TLC(Line(mm).set(1)).rect(4))/TLC(Line(mm).set(2)).width > 4)   
            Line(mm).set(bad_in) = [];
        end
    end
    sws = [TLC(Line(mm).set).sw]; 
    Line(mm).sw = max(sws);
    linerects = reshape([TLC(Line(mm).set).rect]',[4 length(Line(mm).set)])';    
    Line(mm).rect = [min(linerects(:,3)) , min(linerects(:,1))  ,max(linerects(:,4))-min(linerects(:,3))+1, max(linerects(:,2))-  min(linerects(:,1))+1];
    linecolors = reshape([TLC(Line(mm).set).color]',[3 length(Line(mm).set)])';  
    Line(mm).color = mean(linecolors,1);
end

if length(Line) > 0
    linerects = reshape([Line.rect]',[4 length(Line)])'; 
    Line(linerects(:,3) ./ linerects(:,4) <0.45) = [];
    linerects(linerects(:,3) ./ linerects(:,4) <0.45,:) = [];
    [~,ind] = sort(linerects(:,1));
    Line = Line(ind);
    linerects = linerects(ind,:);
else
    linerects = [];
end

%% 5.3. textLine merge
mergeline = zeros(length(Line),1);
for mm = 1:length(Line)
    if mergeline(mm) == 0
        for n = mm+1:length(Line)
            midm = Line(mm).rect(2) + Line(mm).rect(4)/2;
            midn = Line(n).rect(2) + Line(n).rect(4)/2;
            Lv = min(Line(mm).rect(2)+Line(mm).rect(4),Line(n).rect(2)+Line(n).rect(4)) - max(Line(mm).rect(2),Line(n).rect(2));
            Rv = Lv / max(Line(mm).rect(4),Line(n).rect(4));
            hsv1 = rgb2hsv(Line(mm).color/255); hs1 = hsv1(1:2);
            hsv2 = rgb2hsv(Line(n).color/255); hs2 = hsv2(1:2);
            Dc1 = norm(255*abs(hs1-hs2) )/ 1.414;
            Dc2 = norm(abs(Line(mm).color - Line(n).color))/1.732 ; 
            Dc = min(Dc1,Dc2);
            HD1 = abs (Line(mm).ave_height - Line(n).ave_height)/max(Line(mm).ave_height , Line(n).ave_height);
            HD2 = abs (Line(mm).rect(4) - Line(n).rect(4))/max(Line(mm).rect(4) , Line(n).rect(4));
            sw_rate = Line(mm).sw / Line(n).sw;
%                 abs(midm - midn) < max( 20 , 0.2*( Line(m).rect(4) + Line(n).rect(4)))
            if  ((Rv > 0.35 && Line(n).rect(1) - Line(mm).rect(1) > 0.3 *  Line(mm).rect(4)) || Rv > 0.9) ...
                    && abs(midm - midn) < max( 20 , 0.262*( Line(mm).rect(4) + Line(n).rect(4)))...
                    && ( HD1 < 0.3 | HD2 < 0.223 ) && Dc < 38.3 && abs(Line(mm).ave_width - Line(n).ave_width) < 30 ...
                    && Line(n).rect(1) - Line(mm).rect(1)-  Line(mm).rect(3) < 1.7 * (Line(mm).ave_height + Line(n).ave_height) ...
                    && sw_rate > 0.4 && sw_rate < 2.5
                Line(mm).rect = [min(Line(mm).rect(1),Line(n).rect(1)),min(Line(mm).rect(2),Line(n).rect(2)),...
                    max(Line(mm).rect(1)+Line(mm).rect(3),Line(n).rect(1)+Line(n).rect(3)),max(Line(mm).rect(2)+Line(mm).rect(4),Line(n).rect(2)+Line(n).rect(4))];
                Line(mm).rect(3) = Line(mm).rect(3) - Line(mm).rect(1) + 1;
                Line(mm).rect(4) = Line(mm).rect(4) - Line(mm).rect(2) + 1;
                Line(mm).len = Line(mm).len + Line(n).len;
                Line(mm).set = [Line(mm).set  Line(n).set];
                Line(mm).ave_width = (Line(mm).ave_width + Line(n).ave_width) /2;
                Line(mm).ave_gap = (Line(mm).ave_gap + Line(n).ave_gap) /2;
                Line(mm).color = (Line(mm).color + Line(n).color) /2;
                mergeline(n) = 1;
            end
        end
    end
end
Line(mergeline == 1) = [];


linerects = reshape([Line.rect]',[4 length(Line)])'; 
RGB3 = insertObjectAnnotation(rgb, 'rectangle', linerects, 1:length(Line),'Color', 'yellow', 'TextColor', 'black');




%% 6. Remove False textline  
%% 6.1 remove those small line, inside line
% remove those inside Line,with small length or centered
if length(Line) > 0
    linerects = reshape([Line.rect]',[4 length(Line)])'; 
    ws = linerects(:,3) ;
    hs = linerects(:,4);
    linerects(:,3) = linerects(:,3) + linerects(:,1) -1;
    linerects(:,4) = linerects(:,4) + linerects(:,2) -1;
    lens = [Line.len];
    duplica = zeros(length(Line),1);
    for ll = 1:length(Line)
        if max(ws(ll),hs(ll)) < min(25,0.035 * min(m,n))
            duplica(ll) = 1;
        end
        if duplica(ll) == 0 && lens(ll) > 1 && hs(ll) < 0.43 * m
            dif = linerects - repmat(linerects(ll,:),length(Line),1); 
            pos = find(  dif(:,1)>-8 & dif(:,2)>-8 & dif(:,3)<8 & dif(:,4)<14);
            duplica(pos(pos~=ll)) =1;
        elseif duplica(ll) == 0 && lens(ll) == 1 && hs(ll) < 0.43 * m
            dif = linerects - repmat(linerects(ll,:),length(Line),1); 
            pos = find(  dif(:,1)>-8 & dif(:,2)>-8 & dif(:,3)<8 & dif(:,4)<8 & lens' == 1);
            duplica(pos(pos~=ll)) =1;
        elseif duplica(ll) == 0 && lens(ll) > 1
            dif = linerects - repmat(linerects(ll,:),length(Line),1); 
            pos = find(  dif(:,1)>-3 & dif(:,2)>-3 & dif(:,3)<3 & dif(:,4)<3 & ...
                ~( lens' > 2 &  ( ( linerects(:,2) -linerects(ll,2) < 0.2 * hs(ll) & linerects(:,2) >= linerects(ll,2) & linerects(:,4)  < linerects(ll,2) + 0.5 * hs(ll) ) | ...
                                   ( linerects(ll,4) -linerects(:,4) < 0.3 * hs(ll) & linerects(ll,4)>= linerects(:,4) & linerects(:,2)  > linerects(ll,2) + 0.5 * hs(ll) )  )...
                 )  & lens' < 5 );
            duplica(pos(pos~=ll)) =1;
        end
    end
    Line(duplica == 1) = [];         
    linerects = reshape([Line.rect]',[4 length(Line)])'; 
else
    linerects = [];
end

%% 6.2 Entropy model
goodText = zeros(length(Line),1);
if ~isempty(Line)
    for ll = 1:length(Line)
        labels = [TLC(Line(ll).set).label]; 
        gnumbers = labels(labels<10 & labels>1);
        numbers = labels(labels<10);
        clabels = labels(labels<75);
        ulabels = unique(clabels);
        his = hist(clabels,75);
        his = his(his~=0);
        his = his / sum(his);
        rentro = sum(-log(his) .* his)/log(2);
        if length(labels) <= 1
            entropy(ll) = 0;
        else
            entropy(ll) =  rentro  *  (1-sum(labels==75)/length(labels)) ;
        end
        if length(gnumbers) > 1 || length(ulabels) > 3 || labels(1) == 11 || (length(numbers) == length(labels) && length(numbers)>1 &&length(gnumbers)>0 )
            goodText(ll) = 1;
        end
    end
end

if ~isempty(linerects)
    linerects(:,3) = linerects(:,1) + linerects(:,3) - 1;
    linerects(:,4) = linerects(:,2) + linerects(:,4) - 1;
end

%% 6.3 sliding based CNN + Entropy
nnet = cudaconvnet_to_mconvnet('3rd_party_tools/detnet_layers.mat');
[p,q] = size(linerects);
imgs = [];
patch_offset = zeros(p,1);
for i = 1:p      
    pad_up = max(5,floor(0.051 *( linerects(i,4) - linerects(i,2)+1) + 0.5));
    pad_left = max(5,floor(0.065 *( linerects(i,3) - linerects(i,1)+1) + 0.5));
    left = max(1,linerects(i,1)-pad_left);
    right = min(size(rgb,2),linerects(i,3)+pad_left);
    up = max(1,linerects (i,2)-pad_up);
    down = min(size(rgb,1),linerects(i,4)+pad_up);
%     linerects(i,:) = [left up right down];
    im = rgb( up:down ,left:right , : );  
    img = single(rgb2gray(im));
    [mp,np] = size(img);
    img = imresize(img,[24 floor(np/mp*24)]);
    [mp,np] = size(img);
    win_width = floor(mp*1);
    step = floor(mp*0.1);
    if(np<1.3 * win_width)
        patch_num = 1;
        img_batch = imresize(img,[24 24]);
        imgs = cat(4, imgs, img_batch);
    else
        patch_num = length(1:step:np-win_width+1);
        for j = 1:step:np-win_width+1;
            img_batch = img(:,j:(j+win_width-1));
            imgs = cat(4, imgs, img_batch);
        end
    end
    if i == 1
        patch_offset(i) = patch_num;
    else
        patch_offset(i) = patch_offset(i-1)+ patch_num;
    end
end
% data normalization
data = reshape(imgs, [], size(imgs,4));
mu = mean(data, 1);
data = data - repmat(mu, size(data,1), 1);
v = std(data, 0, 1);
data = data ./ (0.0000001 + repmat(v, size(data,1), 1));
imgs = reshape(data, size(imgs));
clear data;
 % load model
nn = nnet;
nn = nn.forward(nn, struct('data', single(imgs)));
[convidence,pred] = max(squeeze(nn.Xout(:,:,1:48,:)), [], 1);
result_label = zeros(p,1);
for i = 1:p
    if i == 1
        cato = pred(1:patch_offset(i));
        hists = convidence(1:patch_offset(i));
    else
        cato = pred(patch_offset(i-1)+1:patch_offset(i)); 
        hists = convidence(patch_offset(i-1)+1:patch_offset(i)); 
    end
    [~,index] = max(hists);
    if cato(index) > 1 && entropy(p) >= 1
        result_label(i) = 1;
    elseif (sum(hists(cato == 1))   > 0.8*sum(hists(cato > 1))  && entropy(p) > 1) || (sum(hists(cato == 1))   > 0.4*sum(hists(cato > 1))  && entropy(p) <= 1)
        result_label(i) = 4;
    else
        result_label(i) = 1;
    end
    wid = linerects(i,3)- linerects(i,1) + 1;
    hei = linerects(i,4)- linerects(i,2) + 1;
    if 0.35*hei>wid || max(wid,hei) < 33
        result_label(i) = 5;
    end
    if length(cato) == 1
        if ( (cato(1) == 26 || cato(1) == 8 ) && hists(1) < 15) || cato(1) == 14 || cato(1) == 15 ||  cato(1) == 19 || (cato(1) == 9 && ( hists(1)< 10.5 ||  max(wid,hei) < 86)) ...
                || cato(1) == 36 ||(cato(1) == 31&& max(wid,hei) <  210)|| cato(1) == 22 || cato(1) ==23  || (cato(1) == 3 && wid / hei > 0.588) ...
                || ( cato(1) == 16 && ( wid > 270 || wid < 97) ) 
            result_label(i) = 5; 
        end
        if hists(1) < 6  || (hists(1)  < 7.3 && max(m,n) < 50 && cato(1) >11 ) || (cato(1) == 20 && (min(hei,wid) < 106 || wid / hei < 0.655 || wid > 235))
            result_label(i) = 4;
        end
    end
end
result_label(goodText==1) = 2;
linerects = linerects(result_label < 4,:);      
if ~isempty(linerects)
    linerects(:,3) = linerects(:,3) - linerects(:,1) + 1;
    linerects(:,4) = linerects(:,4) - linerects(:,2) + 1;
end
RGB3 = insertObjectAnnotation(rgb, 'rectangle', linerects, 1:size(linerects,1) ,'Color', 'green', 'TextColor', 'black');
figure(1)
imshow(rgb);
figure(2)
imshow(RGB3);

toc