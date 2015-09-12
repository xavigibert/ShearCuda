function [pd] = pd_from_pfa(pfa, v_pd, v_pfa)

for idx = 1:length(v_pfa)-1
    if (v_pfa(idx) <= pfa && v_pfa(idx+1) > pfa) || ...
         (v_pfa(idx) >= pfa && v_pfa(idx+1) < pfa)
        pd = v_pd(idx) + (v_pd(idx+1) - v_pd(idx)) * (pfa - v_pfa(idx)) / (v_pfa(idx+1) - v_pfa(idx));
        return;
    end
end
pd = 0;
