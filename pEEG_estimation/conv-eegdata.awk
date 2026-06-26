#!/usr/bin/awk -f   
#
# Usage:  ./conv-eegdata.awk  data-orig/*_waves.csv
#
BEGIN{
   FS=",";
}
  
{  if (FNR==1)
   {  eeg1=eeg2=0;
      last_eeg = 10000000;
      for (i=1; i<=NF; i++)
      {  split($(i), teil, "-");
         if (teil[1]=="Intellivue/EEG_1")  eeg1=i;
         if (teil[1]=="Intellivue/EEG_2")  eeg2=i;
      }
      split(FILENAME,fn1,"/");
      split(fn1[2],fn2,".");
      name="data/" fn2[1];
      print ARGIND-1, fn2[1], start[ARGIND-1]/125, stop[ARGIND-1]/125, eeg1, eeg2;
   }
   else
   {  split($1, ti1, " ");
      split(ti1[2], ti2, ":");
      ti=3600*ti2[1]+60*ti2[2]+ti2[3];
      if ($(eeg1)!="" || $(eeg2)!="")  
      {  ea++;
         if (last_eeg < ti-9)
            for (i=last_eeg+8; i<ti-4; i+=8)
            {  print $1, 300, 300 >name ".eeg";  
               ea++; e0++;
            }
         if ($(eeg1) == "" || $(eeg1) >= 187.5 || $(eeg1) <= -187.5)  e1++;
         if ($(eeg2) == "" || $(eeg2) >= 187.5 || $(eeg2) <= -187.5)  e2++;
         if ($(eeg1) == "")  print $1, 300, $(eeg2) >name ".eeg"
         else if ($(eeg2) == "")  print $1, $(eeg1), 300 >name ".eeg"
         else print $1, $(eeg1), $(eeg2) >name ".eeg"
         last_eeg = ti;
      }
   }
}
ENDFILE{
   if (ARGIND>1)  print  fn2[1], eeg1, eeg2, 100*e0/ea, 100*e1/ea, 100*e2/ea, ea;
   e0=e1=e2=ea=0;
}
