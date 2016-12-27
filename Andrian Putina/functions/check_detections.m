function [specificity, sensitivity, falsealarm, missdetection] = check_detections(previsione, true_class)

    verinegativi = 0;
    falsipositivi = 0;
    veripositivi = 0;
    falsinegativi = 0;
    
    for i = 1 : length(true_class) 
        if (true_class(i) == 1)
            if (true_class(i) - previsione(i) == 0)
                verinegativi = verinegativi + 1;
            else 
                falsipositivi = falsipositivi + 1;
            end
        else
            if (true_class(i) - previsione(i) == 0) 
                veripositivi = veripositivi + 1;
            else 
                falsinegativi = falsinegativi + 1;
            end
        end
    end

    specificity = verinegativi / (verinegativi + falsipositivi); %true negative
    sensitivity = veripositivi / (veripositivi + falsinegativi); % true positive
    falsealarm = falsipositivi / (verinegativi + falsipositivi);
    missdetection = falsinegativi / (veripositivi + falsinegativi);

end