function [th] = th_from_pfa(pfa, v_th, v_pfa)

for idx = 1:length(v_pfa)-1
    if (v_pfa(idx) <= pfa && v_pfa(idx+1) > pfa) || ...
         (v_pfa(idx) >= pfa && v_pfa(idx+1) < pfa)
        th = v_th(idx) + (v_th(idx+1) - v_th(idx)) * (pfa - v_pfa(idx)) / (v_pfa(idx+1) - v_pfa(idx));
        return;
    end
end
th = 0;
