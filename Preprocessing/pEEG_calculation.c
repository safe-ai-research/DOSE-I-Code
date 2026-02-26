/******************************************************************************
*** Program for calculation of pEEG, i.e., parameter time series            *** 
*** from 2-channel EEG data recordings                                      ***
***                                                                         ***
***      by Jan W. Kantelhardt                               14.03.2025     ***
***                                                                         ***
*** Parameters:                                                             ***
*** 1. Duration of band averaging (in multiples of the period, usually 10)  ***
*** 2. Shift of the averaging intervals (same unit as 1., usually 5)        ***
*** 3. EEG maximum value (187 for DOSE-I, 249 for SAVE-AI)                  ***
*** 4.-n. Files containing time series                                      ***
***                                                                         ***
******************************************************************************/

#define  Anzahl_Zeitreihen  2
#define  maxLaenge  524288
#define  log2maxLaenge  19
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
  double r;
  double i;
} complex;

/* Assumptions: 2 channels, at most maxLaenge samples per series */

FILE  *Datei;
double  data[2*maxLaenge], phase1[maxLaenge], phase2[maxLaenge], amp1[maxLaenge], 
        amp2[maxLaenge], gamt[maxLaenge], ampt[maxLaenge*2], MF[maxLaenge/60], SEF95[maxLaenge/60], 
        WSMF30[maxLaenge/60], WSMF49[maxLaenge/60], PE31[maxLaenge/60], PE32[maxLaenge/60], 
        PE61[maxLaenge/60], SFS[maxLaenge/60], CFS[maxLaenge/60], PFS[maxLaenge/60], mws[39], mw2[39],
        MFb[maxLaenge/60], SEF95b[maxLaenge/60], WSMF30b[maxLaenge/60], WSMF49b[maxLaenge/60], 
        WSMFneu[maxLaenge/60], WSMFneub[maxLaenge/60];  
char  date[12*maxLaenge], tim[17*maxLaenge];
int  Anz[maxLaenge/60], Anzb[maxLaenge/60], breite_fakt, intervall_fakt, azt[maxLaenge], azt2[maxLaenge], 
        mwa[24], mwa2, mwa2b, max_EEG, p[65536];
complex  w1[maxLaenge/2], w2[maxLaenge/2], w3[maxLaenge/2], x[maxLaenge], xh[maxLaenge], 
         fourier[2*maxLaenge];

/* Procedures for FFT
 * Procedures
 *   irev....bit reversal (bit mirroring)
 *   fftab...phase factors
 *   fft1d...actual FFT (in 1 dimension)
 * Variables
 *   n=2^m...number of data points
 *   w.......table of phase factors
 *   x.......array of original data or Fourier components
 *   idir....transform direction (+/-1 = forward / inverse)
 * NB: data are overwritten by Fourier components
*/

/**********************************************************************/
int irev(int i,int m) {
/**********************************************************************/
  int ir,k;
  ir=0;
  for(k=1;k<=m;k++) {
    ir=ir|((i>>(k-1))&1)<<(m-k);
  }
  return ir;
}

/**********************************************************************/
void fftab(complex *w,int idir,int n) {
/**********************************************************************/
  int i;
  double arg;
  for(i=0;i<n/2;i++) {
    arg=i*idir*2.0*M_PI/n;
    w[i].r=cos(arg);
    w[i].i=-sin(arg);
  }
}

/**********************************************************************/
void fft1d(complex *x,complex *w,int idir,int n) {
/**********************************************************************/
  int i,i1,i2,inc1,inc2,incn,j,k,l;
  complex z;
  incn=n;
  inc1=1;
  while(incn>1) {
    incn=incn/2;
    inc2=2*inc1;
    i=0;
    for(j=0;j<incn;j++) {
      k=0;
      for(l=0;l<inc1;l++) {
	i1=i+l;
	i2=i1+inc1;
	z.r=w[k].r*x[i2].r-w[k].i*x[i2].i;
	z.i=w[k].r*x[i2].i+w[k].i*x[i2].r;
	x[i2].r=x[i1].r-z.r;
	x[i2].i=x[i1].i-z.i;
	x[i1].r=x[i1].r+z.r;
	x[i1].i=x[i1].i+z.i;
	k=k+incn;
      }
      i=i+inc2;
    }
    inc1=inc2;
  }
  if(idir>0) {
    for(i=0;i<n;i++) {
      x[i].r=x[i].r/n;
      x[i].i=x[i].i/n;
    }
  }
}

/**********************************************************************/
void getphaseamp(int j, int fmin, int fmax, double *phase, double *amp)
/**********************************************************************/
{ int i, ir;
  complex z;

  for(i=0; i<maxLaenge; i++)		/* Hilbert transform: delete all Fourier coefficients */
  {  xh[i].r = x[i].r = 0.0;
     xh[i].i = x[i].i = 0.0;
  }
  for(i=fmin; i<=fmax; i++) 		/* take the selected Fourier coefficients (positive frequencies) */
  {  xh[i].i = -(x[i].r = fourier[j*maxLaenge+i].r);
     xh[i].r = x[i].i = fourier[j*maxLaenge+i].i;
  }
  for(i=maxLaenge-fmax; i<=maxLaenge-fmin; i++) 	/* negative frequencies */
  {  xh[i].i = x[i].r = fourier[j*maxLaenge+i].r;
     xh[i].r = -(x[i].i = fourier[j*maxLaenge+i].i);
  }
  for(i=0; i<maxLaenge; i++)		/* permute data */
  {  ir = irev(i,log2maxLaenge);
     if (ir>i) 
     {  z.r=x[ir].r;
        z.i=x[ir].i;
        x[ir].r=x[i].r;
        x[ir].i=x[i].i;
        x[i].r=z.r; 
        x[i].i=z.i;
        z.r=xh[ir].r;			/* also for the Hilbert transform */
        z.i=xh[ir].i;
        xh[ir].r=xh[i].r;
        xh[ir].i=xh[i].i;
        xh[i].r=z.r;
        xh[i].i=z.i;
     }
  }
  fft1d(x,w2,-1,maxLaenge);		/* inverse FFT */
  fft1d(xh,w2,-1,maxLaenge);

  for(i=0; i<maxLaenge; i++)		/* compute phases and amplitudes */
  {  phase[i] = atan2(xh[i].r, x[i].r);
     amp[i] = sqrt(xh[i].r*xh[i].r + x[i].r*x[i].r);
     if ((fabs(xh[i].i/xh[i].r)>0.05) || (fabs(x[i].i/x[i].r)>0.05))
     {  printf("Symmetry error in the FFT: j=%d, fmin=%d, fmax=%d, f=%d, x.i=%6.3f, xh.i=%6.3f\n", 
           j, fmin, fmax, i, x[i].i/x[i].r, xh[i].i/xh[i].r);
        exit(-1);
     }
  }
}


/**********************************************************************/
void Ausgabe(char *nam, int n, double *var1, double *var2) 
/**********************************************************************/
{ char  Name[100];
  int  i, j, k, l, az;
  double  sum;

  strcpy(Name, nam);  		/* output of the compressed results */
  Name[strlen(Name)-4] = 0;
  strcat(Name, ".erg.csv");
  printf("Writing output file %s\n", Name);
  Datei = fopen(Name, "w");
  fprintf(Datei, "Time,abs_subdelta,abs_delta1,abs_delta2,abs_theta,abs_alpha,abs_beta1,abs_beta2,abs_gamma,rel_subdelta,rel_delta1,rel_delta2,rel_theta,rel_alpha,rel_beta1,rel_beta2,rel_gamma,sync_subdelta,sync_delta1,sync_delta2,sync_theta,sync_alpha,sync_beta1,sync_beta2,sync_gamma,MF,SEF95,WSMF30,WSMF49,PE31,PE32,PE61,SFS,CFS,PFS,WSMF_Klimpel,MF_Jordan,SEF95_Jordan,WSMF30_16,WSMF49_16,WSMF_Klimpel_16\n");
  for (i=10; i<=n/125-10; i++)
  {  fprintf(Datei, "%s %s", date+12*i*125, tim+17*i*125);
     for (l=0; l<16; l++)
     {  sum = az = 0;
        for (j=-3; j<=3; j++) 
           if (var1[16*(i+j) + l] != -9.9999) 
           {  sum += var1[16*(i+j) + l];
              az++;
           }
        if (az>0 && sum>0)
        {  fprintf(Datei, ",%6.4f", sum = log(sum/az));
           mws[l] += sum;
           mw2[l] += sum*sum;
           mwa[l]++;
        }
        else  fprintf(Datei, ",");
     }
     for (l=0; l<8; l++)
     {  sum = az = 0;
        if (i>=10 && i<=n/125-10)
           for (j=-10; j<=10; j++) 
              if (var2[8*(i+j) + l] != -9.9999) 
              {  sum += var2[8*(i+j) + l];
                 az++;
              }
        if (az>0)
        {  fprintf(Datei, ",%6.4f", sum = sum/az);
           mws[l+16] += sum;
           mw2[l+16] += sum*sum;
           mwa[l+16]++;
        }
        else  fprintf(Datei, ",");
     }
     if (Anz[2*i] > 0)
     {  mws[24] += (MF[2*i]/=Anz[2*i]);  mw2[24] += MF[2*i]*MF[2*i];
        mws[25] += (SEF95[2*i]/=Anz[2*i]);  mw2[25] += SEF95[2*i]*SEF95[2*i];
        mws[26] += (WSMF30[2*i]/=Anz[2*i]);  mw2[26] += WSMF30[2*i]*WSMF30[2*i];
        mws[27] += (WSMF49[2*i]/=Anz[2*i]);  mw2[27] += WSMF49[2*i]*WSMF49[2*i];
        mws[28] += (PE31[2*i]/=Anz[2*i]);  mw2[28] += PE31[2*i]*PE31[2*i];
        mws[29] += (PE32[2*i]/=Anz[2*i]);  mw2[29] += PE32[2*i]*PE32[2*i];
        mws[30] += (PE61[2*i]/=Anz[2*i]);  mw2[30] += PE61[2*i]*PE61[2*i];
        mws[31] += (SFS[2*i]/=Anz[2*i]);  mw2[31] += SFS[2*i]*SFS[2*i];
        mws[32] += (CFS[2*i]/=Anz[2*i]);  mw2[32] += CFS[2*i]*CFS[2*i];
        mws[33] += (PFS[2*i]/=Anz[2*i]);  mw2[33] += PFS[2*i]*PFS[2*i];
        mws[34] += (WSMFneu[2*i]/=Anz[2*i]);  mw2[34] += WSMFneu[2*i]*WSMFneu[2*i];
        mwa2++;
     }
     if (Anz[2*i+1] > 0)
     {  mws[24] += (MF[2*i+1]/=Anz[2*i+1]);  mw2[24] += MF[2*i+1]*MF[2*i+1];
        mws[25] += (SEF95[2*i+1]/=Anz[2*i+1]);  mw2[25] += SEF95[2*i+1]*SEF95[2*i+1];
        mws[26] += (WSMF30[2*i+1]/=Anz[2*i+1]);  mw2[26] += WSMF30[2*i+1]*WSMF30[2*i+1];
        mws[27] += (WSMF49[2*i+1]/=Anz[2*i+1]);  mw2[27] += WSMF49[2*i+1]*WSMF49[2*i+1];
        mws[28] += (PE31[2*i+1]/=Anz[2*i+1]);  mw2[28] += PE31[2*i+1]*PE31[2*i+1];
        mws[29] += (PE32[2*i+1]/=Anz[2*i+1]);  mw2[29] += PE32[2*i+1]*PE32[2*i+1];
        mws[30] += (PE61[2*i+1]/=Anz[2*i+1]);  mw2[30] += PE61[2*i+1]*PE61[2*i+1];
        mws[31] += (SFS[2*i+1]/=Anz[2*i+1]);  mw2[31] += SFS[2*i+1]*SFS[2*i+1];
        mws[32] += (CFS[2*i+1]/=Anz[2*i+1]);  mw2[32] += CFS[2*i+1]*CFS[2*i+1];
        mws[33] += (PFS[2*i+1]/=Anz[2*i+1]);  mw2[33] += PFS[2*i+1]*PFS[2*i+1];
        mws[34] += (WSMFneu[2*i]/=Anz[2*i+1]);  mw2[34] += WSMFneu[2*i+1]*WSMFneu[2*i+1];
        mwa2++;
     }
     if (Anz[2*i] > 0 && Anz[2*i+1] > 0)
        fprintf(Datei, ",%5.3f,%5.3f,%5.3f,%5.3f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%5.3f", 
           (MF[2*i]+MF[2*i+1])/2, (SEF95[2*i]+SEF95[2*i+1])/2, (WSMF30[2*i]+WSMF30[2*i+1])/2, 
           (WSMF49[2*i]+WSMF49[2*i+1])/2, (PE31[2*i]+PE31[2*i+1])/2, (PE32[2*i]+PE32[2*i+1])/2, 
           (PE61[2*i]+PE61[2*i+1])/2, (SFS[2*i]+SFS[2*i+1])/2, (CFS[2*i]+CFS[2*i+1])/2, 
           (PFS[2*i]+PFS[2*i+1])/2, (WSMFneu[2*i]+WSMFneu[2*i+1])/2); 
     else if (Anz[2*i] > 0)
        fprintf(Datei, ",%5.3f,%5.3f,%5.3f,%5.3f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%5.3f", 
           MF[2*i], SEF95[2*i], WSMF30[2*i], WSMF49[2*i], PE31[2*i], PE32[2*i], PE61[2*i], 
           SFS[2*i], CFS[2*i], PFS[2*i], WSMFneu[2*i]);  
     else if (Anz[2*i+1] > 0)
        fprintf(Datei, ",%5.3f,%5.3f,%5.3f,%5.3f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%6.4f,%5.3f", 
           MF[2*i+1], SEF95[2*i+1], WSMF30[2*i+1], WSMF49[2*i+1], PE31[2*i+1], PE32[2*i+1], PE61[2*i+1], 
              SFS[2*i+1], CFS[2*i+1], PFS[2*i+1], WSMFneu[2*i+1]);
     else
        fprintf(Datei, ",,,,,,,,,,,");

     if (Anzb[2*i] > 0)
     {  mws[35] += (MFb[2*i]/=Anzb[2*i]);  mw2[35] += MFb[2*i]*MFb[2*i];
        mws[36] += (SEF95b[2*i]/=Anzb[2*i]);  mw2[36] += SEF95b[2*i]*SEF95b[2*i];
        mws[37] += (WSMF30b[2*i]/=Anzb[2*i]);  mw2[37] += WSMF30b[2*i]*WSMF30b[2*i];
        mws[38] += (WSMF49b[2*i]/=Anzb[2*i]);  mw2[38] += WSMF49b[2*i]*WSMF49b[2*i];
        mws[39] += (WSMFneub[2*i]/=Anzb[2*i]);  mw2[39] += WSMFneub[2*i]*WSMFneub[2*i];
        mwa2b++;
     }
     if (Anzb[2*i+1] > 0)
     {  mws[35] += (MFb[2*i+1]/=Anzb[2*i+1]);  mw2[35] += MFb[2*i+1]*MFb[2*i+1];
        mws[36] += (SEF95b[2*i+1]/=Anzb[2*i+1]);  mw2[36] += SEF95b[2*i+1]*SEF95b[2*i+1];
        mws[37] += (WSMF30b[2*i+1]/=Anzb[2*i+1]);  mw2[37] += WSMF30b[2*i+1]*WSMF30b[2*i+1];
        mws[38] += (WSMF49b[2*i+1]/=Anzb[2*i+1]);  mw2[38] += WSMF49b[2*i]*WSMF49b[2*i+1];
        mws[39] += (WSMFneub[2*i+1]/=Anzb[2*i+1]);  mw2[39] += WSMFneub[2*i+1]*WSMFneub[2*i+1];
        mwa2b++;
     }
     if (Anzb[2*i] > 0 && Anzb[2*i+1] > 0)
        fprintf(Datei, ",%5.3f,%5.3f,%5.3f,%5.3f,%5.3f\n", 
           (MFb[2*i]+MFb[2*i+1])/2, (SEF95b[2*i]+SEF95b[2*i+1])/2, (WSMF30b[2*i]+WSMF30b[2*i+1])/2, 
           (WSMF49b[2*i]+WSMF49b[2*i+1])/2, (WSMFneub[2*i]+WSMFneub[2*i+1])/2); 
     else if (Anzb[2*i] > 0)
        fprintf(Datei, ",%5.3f,%5.3f,%5.3f,%5.3f,%5.3f\n", 
           MFb[2*i], SEF95b[2*i], WSMF30b[2*i], WSMF49b[2*i], WSMFneub[2*i]);  
     else if (Anzb[2*i+1] > 0)
        fprintf(Datei, ",%5.3f,%5.3f,%5.3f,%5.3f,%5.3f\n", 
           MFb[2*i+1], SEF95b[2*i+1], WSMF30b[2*i+1], WSMF49b[2*i+1], WSMFneub[2*i+1]);
     else
        fprintf(Datei, ",,,,,\n");
  }
  fclose(Datei);
}

/**********************************************************************/
void  WSMF(double *res, int f1, int f2, double p, double r)
/**********************************************************************/
{  int i;
   double sum1, sum2;
   sum1=sum2=0.;
   for (i=f1; i<=f2; i++) 
      sum1 += pow(x[i].r*x[i].r + x[i].i*x[i].i, p/2);
   sum1*=r;
   for (i=f1; sum2<sum1; i++)
      sum2 += pow(x[i].r*x[i].r + x[i].i*x[i].i, p/2);
   *res = 125*i/1024.;
}

/**********************************************************************/
void  WSMF2(double *res, int f1, int f2, double p, double r)
/**********************************************************************/
{  int i;
   double sum1, sum2;
   sum1=sum2=0.;
   for (i=f1; i<=f2; i++) 
      sum1 += pow(x[i].r*x[i].r + x[i].i*x[i].i, p/2);
   sum1*=r;
   for (i=f1; sum2<sum1; i++)
      sum2 += pow(x[i].r*x[i].r + x[i].i*x[i].i, p/2);
   *res = 125*i/2048.;
}


/**********************************************************************/
void  PEntr(double *res, int *a, int n, int tau, double s)
/**********************************************************************/
{  int i, j, j1, m, m1, pa;
   double  sum, pr;
   
   for (i=0; i<65536; i++)  p[i] = 0;
   pa=0;
   for (i=0; i<1024-(n-1)*tau; i++)
   {  m=m1=0;
      for (j=0; j<n-1; j++)
         for (j1=j+1; j1<n; j1++)
         {  m = m<<1;
            if (x[i+j1*tau].r > x[i+j*tau].r +0.5) 
               m+=1;
            else if (x[i+j1*tau].r >= x[i+j*tau].r -0.5 && n<5)
            {  m1=1; j=j1=n;  }
         }
      if (m1>0)  p[10000]++;
      else  p[m]++;
      pa++;
   }
   sum = 0;
   for (i=0; i<65536; i++)
      if (p[i] > 0)  
      {  pr = p[i];  pr /= pa;
         sum+=pr*log(pr);
         (*a)++;
      }
   *res = -sum;
}

/**********************************************************************/
void  Bispectrum(double *SFS, double *CFS, double *PFS, int f1, int f2, int f3, int f4)  
/**********************************************************************/
{  int  i, i1, j, j1, c2;
   double  sr, si, s1, s2, s3, t, rtp1, rtp2, bic1, bic2, pow1, pow2;
  
   rtp1 = rtp2 = bic1 = bic2 = pow1 = pow2 = 0.;
   c2 = 0;
   for (i=f1; i+7<=f2; i+=8)
      for (j=f1; j<=i && i+j+14<=511; j+=8)
      {  sr = si = s1 = s2 = s3 = 0.;
         for (i1=0; i1<8; i1++)
         {  s1 += x[i+i1].r * x[i+i1].r + x[i+i1].i * x[i+i1].i;
            for (j1=0; j1<8; j1++)
            {  sr += x[i+i1].r*x[j+j1].r*x[i+i1+j+j1].r - x[i+i1].i*x[j+j1].i*x[i+i1+j+j1].r 
                     + x[i+i1].i*x[j+j1].r*x[i+i1+j+j1].i + x[i+i1].r*x[j+j1].i*x[i+i1+j+j1].i;
               si += x[i+i1].i*x[j+j1].r*x[i+i1+j+j1].r + x[i+i1].r*x[j+j1].i*x[i+i1+j+j1].r 
                     - x[i+i1].r*x[j+j1].r*x[i+i1+j+j1].i + x[i+i1].i*x[j+j1].i*x[i+i1+j+j1].i;
               s3 += x[i+i1+j+j1].r * x[i+i1+j+j1].r + x[i+i1+j+j1].i * x[i+i1+j+j1].i;
            }
         }
         for (j1=0; j1<8; j1++)
            s2 += x[j+j1].r * x[j+j1].r + x[j+j1].i * x[j+j1].i;
         if (i==j)
         {  rtp1 += (t = sr*sr + si*si) /4096.;  
            bic1 += t / (s1 * s2 * s3);
            pow1 += s1 / 8.;
         }
         else
         {  rtp1 += (t = sr*sr + si*si) /2048.;  
            bic1 += 2. * t / (s1 * s2 * s3);
            pow1 += s1 / 4.;
         }
      }
   for (i=f3; i+7<=f4; i+=8)
      for (j=f1; j<=i && i+j+14<=511; j+=8)
      {  sr = si = s1 = s2 = s3 = 0.;
         for (i1=0; i1<8; i1++)
         {  s1 += x[i+i1].r * x[i+i1].r + x[i+i1].i * x[i+i1].i;
            for (j1=0; j1<8; j1++)
            {  sr += x[i+i1].r*x[j+j1].r*x[i+i1+j+j1].r - x[i+i1].i*x[j+j1].i*x[i+i1+j+j1].r 
                     + x[i+i1].i*x[j+j1].r*x[i+i1+j+j1].i + x[i+i1].r*x[j+j1].i*x[i+i1+j+j1].i;
               si += x[i+i1].i*x[j+j1].r*x[i+i1+j+j1].r + x[i+i1].r*x[j+j1].i*x[i+i1+j+j1].r 
                     - x[i+i1].r*x[j+j1].r*x[i+i1+j+j1].i + x[i+i1].i*x[j+j1].i*x[i+i1+j+j1].i;
               s3 += x[i+i1+j+j1].r * x[i+i1+j+j1].r + x[i+i1+j+j1].i * x[i+i1+j+j1].i;
            }
         }
         for (j1=0; j1<8; j1++)
            s2 += x[j+j1].r * x[j+j1].r + x[j+j1].i * x[j+j1].i;
         if (i==j)
         {  rtp2 += (t = sr*sr + si*si) /4096.;  
            bic2 += t / (s1 * s2 * s3);
            pow2 += s1 / 8.;
            c2++;
         }
         else
         {  rtp2 += (t = sr*sr + si*si) /2048.;  
            bic2 += 2. * t / (s1 * s2 * s3);
            pow2 += s1 / 4.;
            c2+=2;
         }
      }
   if (rtp2>0)  *SFS = log(rtp2/(rtp1+rtp2))/log(10.);
   else  *SFS = -9.9999;
   if (bic2>0)  *CFS = log(bic2/(bic1+bic2))/log(10.);
   else  *CFS = -9.9999;
   if (pow2>0)  *PFS = log(pow2/(pow1+pow2))/log(10.);
   else  *PFS = -9.9999;
/*   printf("%3d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n", c2, rtp1, rtp2, bic1, bic2, pow1, pow2); */
 }

/**********************************************************************/
int  FFT(double *data, complex *w, int m, int n, int nlog2)
/**********************************************************************/
{  int  ir, k, n1;
   double  mw, mw1;
   complex  z;

   n1=1<<nlog2;
   mw = 0;
   for (ir=0; ir<n; ir++)	/* compute means and subtract them */
      if (data[m*ir] >= -max_EEG-0.49 && data[m*ir] <= max_EEG+0.49)
         mw += data[m*ir];
      else  break;
   if (ir==n)
   {  mw1 = mw/n;
      for (ir=0; ir<n; ir++)
      {  x[ir].r = data[m*ir] - mw1;
         x[ir].i = 0;
      }
      for (ir=n; ir<n1; ir++)
      {  x[ir].r = 0;
         x[ir].i = 0;
      }
      for(ir=0; ir<n1; ir++) 	/* permute original data */
      {  k = irev(ir,nlog2);
         if (k>ir) 
         {  z.r=x[k].r;
            z.i=x[k].i;
            x[k].r=x[ir].r;
            x[k].i=x[ir].i;
            x[ir].r=z.r;
            x[ir].i=z.i;
         }
      }
      fft1d(x,w,1,n1);		/* FFT (overwrites original data) */
      return n1;
   }
   else  return 0;
}        


/**********************************************************************/
int main(int argc, char *argv[]) {
/**********************************************************************/
  int  i, ir, j, k, l, n, n1, m, f, breite, intervall, ind1, ind2, amp, time,
       PEa31, PEa32, PEa61, PEa;
  double  mw, mw1, tmp, re, im, pi = 3.1415962,
  fmin[8] = {0.5, 1.0, 2.0, 4.0,  7.5, 13.0, 20.0, 30.0},
  fmax[8] = {1.0, 2.0, 4.0, 7.5, 13.0, 20.0, 30.0, 49.0};
/* delta0, delta1, delta2, theta, alpha, beta-low, beta-high, gamma */
  complex  z;

  breite_fakt = atoi(argv[1]);
  intervall_fakt = atoi(argv[2]);
  max_EEG = atoi(argv[3]);
  PEa31 = PEa32 = PEa61 = PEa = 0;

for (f=4; f<argc; f++) 
{ printf("File: %s\n", argv[f]);
  m = Anzahl_Zeitreihen;		
  Datei = fopen(argv[f], "r");		/* read data */
  ir=1;
  while (getc(Datei) != '\n');		/* skip the first line */
  for (n=0; (ir>0) && (n<maxLaenge); n++)	/* read usable data until the array is full */
  {  ir = fscanf(Datei, "%s %s ", date+12*n, tim+17*n);
     for (j=0; j<m; j++)
        ir = fscanf(Datei, "%lf", data+m*n+j);
  }
  fclose(Datei);
  n--;
/*  reverse time direction
  for (n1=0; n1<n/2; n1++)
  {  mw = data[m*n1]; data[m*n1] = data[m*(n-n1-1)]; data[m*(n-n1-1)] = mw;  
     mw = data[m*n1+1]; data[m*n1+1] = data[m*(n-n1-1)+1]; data[m*(n-n1-1)+1] = mw;  
  }
*/
  for (i=n; i<maxLaenge; i++)		/* fill up to maxLaenge, boundary condition */
     for (j=0; j<m; j++)
        data[m*i+j] = data[m*(n-1)+j] + (data[j]-data[m*(n-1)+j])*(i-n)/((double)maxLaenge-n);
  printf("Data read: n=%d rows, m=%d columns.\n", n, m);
/*  printf("Start %s %s, End %s %s (%s %s %s %s)\n", date, tim, date+(n-1)*12, tim+(n-1)*17, date+10*125*12, tim+10*125*17, date+(n-10*125)*12, tim+(n-10*125)*17); */

/* data preprocessing, part 1 of the Hilbert transform */
/*   for (j=0; j<m; j++)			compute means and subtract
  {  mw = 0;
     for (i=0; i<n; i++)
        mw += data[m*i+j];
     mw1 = mw/n;
     for (i=0; i<maxLaenge; i++)
        data[m*i+j] -= mw1; 
  }
  printf("Data preprocessing finished.\n"); */

  fftab(w1,1,maxLaenge);		/* initialize phase factors for FFT */
  fftab(w2,-1,maxLaenge);
  for (j=0; j<m; j++)			/* loop over all time series */
  {  for (i=0; i<maxLaenge; i++)
     {  x[i].r = data[m*i+j];
        x[i].i = 0;
     }
     for(i=0; i<maxLaenge; i++) 	/* permute original data */
     {  ir = irev(i,log2maxLaenge);
        if (ir>i) 
        {  z.r=x[ir].r;
           z.i=x[ir].i;
           x[ir].r=x[i].r;
           x[ir].i=x[i].i;
           x[i].r=z.r;
           x[i].i=z.i;
        }
     }
     fft1d(x,w1,1,maxLaenge);		/* FFT (overwrites original data) */

     for(i=0; i<maxLaenge; i++) 	/* store the Fourier coefficients */
     {  fourier[j*maxLaenge+i].r = x[i].r;
        fourier[j*maxLaenge+i].i = x[i].i;
     }
  }
  printf("First FFT finished.\n");

  for (l=0; l<n/125*8; l++)
     ampt[2*l] = ampt[2*l+1] = gamt[l] = azt2[l] = azt[l] = 0;
  for (l=0; l<n/60; l++)  Anz[l] = Anzb[l] = 0;

  for (l=0; l<8; l++)			/* loop over the frequency bands */
  {  breite = breite_fakt/sqrt(fmin[l]*fmax[l])*125;
     intervall = intervall_fakt/10./sqrt(fmin[l]*fmax[l])*125;
     if (intervall == 0)  intervall=1;
     if (n > breite)
     {  printf("Analysis of frequency band from %6d to %6d (index %2d): f = %6.3f\n",
           (int)(fmin[l]*maxLaenge/125), (int)(fmax[l]*maxLaenge/125), l, sqrt(fmin[l]*fmax[l]));
        getphaseamp(0, (int)(fmin[l]*maxLaenge/125), (int)(fmax[l]*maxLaenge/125), phase1, amp1);
        getphaseamp(1, (int)(fmin[l]*maxLaenge/125), (int)(fmax[l]*maxLaenge/125), phase2, amp2);
        for (i=0; i<n-breite; i+=intervall)
        {  time = 8*(int)((i+breite/2.+62.5)/125.);
           amp = 0.0; 
           for (ir=i; ir<i+breite; ir++)		/* amplitude */
              if (data[m*ir] >= -max_EEG-0.49 && data[m*ir] <= max_EEG+0.49)  
                 amp += amp1[ir];
              else  break;
           if (ir == i+breite)
           {  ampt[2*time + l] += amp;
              azt2[time + l]++;
           }
           amp = 0.0; 
           for (ir=i; ir<i+breite; ir++)		/* amplitude */
              if (data[m*ir+1] >= -max_EEG-0.49 && data[m*ir+1] <= max_EEG+0.49)  
                 amp += amp2[ir];
              else  break;
           if (ir == i+breite)
           {  ampt[2*time + l] += amp;
              azt2[time + l]++;
           }
           re = im = 0.0; 
           for (ir=i; ir<i+breite; ir++)		/* synchronization */
              if (data[m*ir] >= -max_EEG-0.49 && data[m*ir] <= max_EEG+0.49 && data[m*ir+1] >= -max_EEG-0.49 && data[m*ir+1] <= max_EEG+0.49)
              {  mw = phase1[ir] - phase2[ir];
                 re += cos(mw);
                 im += sin(mw);
              }
              else  break;
           if (ir == i+breite)
           {  tmp = sqrt(re*re + im*im)/breite;
              gamt[time + l] += tmp;
              azt[time + l]++;
           }
        }
        for (i=0; i<=n/125; i++)
        {  if (azt[8*i + l] > 1)
              gamt[8*i + l] /= azt[8*i + l];
           else 
              gamt[8*i + l] = -9.9999;
           if (azt2[8*i + l] > 1)
              ampt[16*i + l] /= 1000*azt2[8*i + l];
           else
              ampt[16*i + l] = -9.9999;
        }
     }
  }
  for (i=0; i<=n/125; i++)
  {  amp=0; 
     for (l=0; l<8; l++)
        if (ampt[16*i + l] != -9.9999)
           amp += ampt[16*i + l];
     for (l=0; l<8; l++)
        if (ampt[16*i + l] != -9.9999 && amp>0)
           ampt[16*i + 8+l] = ampt[16*i + l]/amp;
        else
           ampt[16*i + 8+l] = -9.9999;
  }

  printf("First part finished.\n");

/* second data analysis (MF, SEF95, WSMF-30 and WSMF-49, bispectrum) */
  fftab(w1,1,1024);			/* initialize phase factors for FFT */
  fftab(w2,-1,1024);
  fftab(w3,1,2048);
  for (i=0; i<n; i+=125)		/* loop over time in seconds */
  {  for (j=0; j<m; j++)		/* loop over all time series */
     {  if (FFT(data+m*i+j, w1, m, 1<<10, 10) >0)
        {  Anz[2*i/125+8+j]++;
           WSMF(MF+2*i/125+8+j,      4, 246, 2.0, 0.5 );
           WSMF(SEF95+2*i/125+8+j,   4, 246, 2.0, 0.95);
           WSMF(WSMF30+2*i/125+8+j, 66, 246, 0.4, 0.5 );
           WSMF(WSMF49+2*i/125+8+j, 66, 401, 1.0, 0.5 );
           WSMF(WSMFneu+2*i/125+8+j, 70, 225, 0.41, 0.32 );

           Bispectrum(SFS+2*i/125+8+j, CFS+2*i/125+8+j, PFS+2*i/125+8+j, 9, 329, 330, 386); 

           for (ir=0; ir<4; ir++)   { x[ir].r = 0; x[ir].i = 0; }	/* filter 0.5 to 45 Hz */
           for (ir=370; ir<655; ir++)   { x[ir].r = 0; x[ir].i = 0; }
           for (ir=1021; ir<1024; ir++)   { x[ir].r = 0; x[ir].i = 0; }
       
           for(ir=0; ir<1024; ir++) 	/* permute original data */
           {  k = irev(ir,10);
              if (k>ir) 
              {  z.r=x[k].r;
                 z.i=x[k].i;
                 x[k].r=x[ir].r;
                 x[k].i=x[ir].i;
                 x[ir].r=z.r;
                 x[ir].i=z.i;
              }
           }
           fft1d(x,w2,-1,1024);		/* inverse FFT (overwrites original data) */

           PEntr(PE31+2*i/125+8+j, &PEa31, 3, 1, 0.5);
           PEntr(PE32+2*i/125+8+j, &PEa32, 3, 2, 0.5);
           PEntr(PE61+2*i/125+8+j, &PEa61, 6, 1, 0.5);
           PEa++;
        }
        if (FFT(data+m*i+j, w3, m, 1<<11, 11) >0)
        {  Anzb[2*i/125+16+j]++;
           WSMF2(MFb+2*i/125+16+j,      8, 491, 2.0, 0.5 );
           WSMF2(SEF95b+2*i/125+16+j,   8, 491, 2.0, 0.95);
           WSMF2(WSMF30b+2*i/125+16+j, 132, 491, 0.4, 0.5 );
           WSMF2(WSMF49b+2*i/125+16+j, 132, 802, 1.0, 0.5 );
           WSMF2(WSMFneub+2*i/125+16+j, 140, 450, 0.41, 0.32 ); 
        }
     }
  }
  printf("Second part finished.\n");

  Ausgabe(argv[f], n, ampt, gamt);
}
  Datei = fopen("means_pEEG.csv", "w");
  for (i=0; i<24; i++)
     fprintf(Datei, "%2d,%7.4f,%6.4f\n", i, mws[i]/mwa[i], sqrt(mwa[i]*mw2[i]-mws[i]*mws[i])/mwa[i]);
  for (i=24; i<35; i++)
     fprintf(Datei, "%2d,%7.4f,%6.4f\n", i, mws[i]/mwa2, sqrt(mwa2*mw2[i]-mws[i]*mws[i])/mwa2);
  for (i=35; i<39; i++)
     fprintf(Datei, "%2d,%7.4f,%6.4f\n", i, mws[i]/mwa2b, sqrt(mwa2b*mw2[i]-mws[i]*mws[i])/mwa2b);
  fclose(Datei);
}



