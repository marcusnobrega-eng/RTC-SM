function [HII] = Human_instability_index(m,L,Cd_HI,u,B,pperson,pfluid,h_max_channel,L_channel,out_r_final,zzz,g)
%% Input from MATLAB
 out_r = out_r_final';
 %zzz is the document's name of the water level series
 A = L_channel.*zzz; %m²
 v = out_r./(A); %m/s

 %% Equation of human instability (Rotava, 2013)
 Fperson = m*g;
 Vc = (m.*zzz)/(L*pperson); 
 Fbuoy = pfluid.*Vc*g;
 Ffriction = u*(Fperson-Fbuoy);
 Fflow = 0.5*pfluid*Cd_HI*B*(v.^2).*zzz;
 HII = Fflow-Ffriction;

end