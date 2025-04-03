# define the VIs to be caluclated while running the script below (has to match function names further down)

import numpy as np
VI=["R","G","B","GCC","RCC","ExG","GLI"]
#All RGB 
#VI=["R","G","B",'BCC','DSWI4','ExG','ExGR','ExR','GCC','GLI','IKAW','MGRVI','MRBVI','NDYI','RCC','RGBVI','RGRI','RI','SI','TGI','VARI','VIG']

# Functions to calculate  VIs, avaliable bands are # Bands RB (red) GB (green) BB (blue) and NB (near infrared)
# for inspiration see https://github.com/awesome-spectral-indices/awesome-spectral-indices/blob/main/output/spectral-indices-table.csv
# Original Bands
def R():	

	return RB

def G():
	return GB

def B():
	return BB

def N():
	return NB

#Radial Growth Phenology Index (Eitel 2023)
def RGPI():
	return (NB-BB)*(NB-GR)*(NB-RB)

#ALL RGB VEGETATION FROM  spectral-indices-table.csv
#Advanced Vegetation Index
def AVI():
	return (NB * (1.0 - RB) * (NB - RB)) ** (1/3)

#Blue Chromatic Coordinate
def BCC():
	return BB / (RB + GB + BB)

#Blue Normalized Difference Vegetation Index
def BNDVI():
	return (NB - BB)/(NB + BB)

#Chlorophyll Index Green
def CIG():
	return (NB / GB) - 1.0

#Chlorophyll Vegetation Index
def CVI():
	return (NB * RB) / (GB ** 2.0)

#Disease-Water Stress Index 4
def DSWI4():
	return GB/RB

#Difference Vegetation Index
def DVI():
	return NB - RB

#Enhanced Normalized Difference Vegetation Index
def ENDVI():
	return ((NB + GB) - (2 * BB)) / ((NB + GB) + (2 * BB))

#Enhanced Vegetation Index of Vegetation
def EVIv():
	return 2.5 * ((NB - RB)/(NB + 6 * RB - 7.5 * BB + 1.0)) * NB

#Excess Green Index
def ExG():
	return 2 * GB - RB - BB

#ExG - ExR Vegetation Index
def ExGR():
	return (2.0 * GB - RB - BB) - (1.3 * RB - GB)

#Excess Red Index
def ExR():
	return 1.3 * RB - GB

#Fluorescence Correction Vegetation Index
def FCVI():
	return NB - ((RB + GB + BB)/3.0)

#Green Atmospherically Resistant Vegetation Index
def GARI():
	return (NB - (GB - (BB - RB))) / (NB - (GB + (BB - RB)))

#Green-Blue Normalized Difference Vegetation Index
def GBNDVI():
	return (NB - (GB + BB))/(NB + (GB + BB))

#Green Chromatic Coordinate
def GCC():
	return GB / (RB + GB + BB)

#Global Environment Monitoring Index
def GEMI():
	return ((2.0*((NB ** 2.0)-(RB ** 2.0)) + 1.5*NB + 0.5*RB)/(NB + RB + 0.5))*(1.0 - 0.25*((2.0 * ((NB ** 2.0) - (RB ** 2)) + 1.5 * NB + 0.5 * RB)/(NB + RB + 0.5)))-((RB - 0.125)/(1 - RB))

#Green Leaf Index
def GLI():
	return (2.0 * GB - RB - BB) / (2.0 * GB + RB + BB)

#Green Normalized Difference Vegetation Index
def GNDVI():
	return (NB - GB)/(NB + GB)

#Green Optimized Soil Adjusted Vegetation Index
def GOSAVI():
	return (NB - GB) / (NB + GB + 0.16)

#Green-Red Normalized Difference Vegetation Index
def GRNDVI():
	return (NB - (GB + RB))/(NB + (GB + RB))

#Green Ratio Vegetation Index
def GRVI():
	return NB/GB

#Kawashima Index
def IKAW():
	return (RB - BB)/(RB + BB)

#Infrared Percentage Vegetation Index
def IPVI():
	return NB/(NB + RB)

#Modified Chlorophyll Absorption in Reflectance Index 1
def MCARI1():
	return 1.2 * (2.5 * (NB - RB) - 1.3 * (NB - GB))

#Modified Chlorophyll Absorption in Reflectance Index 2
def MCARI2():
	return (1.5 * (2.5 * (NB - RB) - 1.3 * (NB - GB))) / ((((2.0 * NB + 1) ** 2) - (6.0 * NB - 5 * (RB ** 0.5)) - 0.5) ** 0.5)

#Modified Green Red Vegetation Index
def MGRVI():
	return (GB ** 2.0 - RB ** 2.0) / (GB ** 2.0 + RB ** 2.0)

#Modified Red Blue Vegetation Index
def MRBVI():
	return (RB ** 2.0 - BB ** 2.0)/(RB ** 2.0 + BB ** 2.0)

#Modified Soil-Adjusted Vegetation Index
def MSAVI():
	return 0.5 * (2.0 * NB + 1 - (((2 * NB + 1) ** 2) - 8 * (NB - RB)) ** 0.5)

#Modified Simple Ratio
def MSR():
	return (NB / RB - 1) / ((NB / RB + 1) ** 0.5)

#Modified Triangular Vegetation Index 1
def MTVI1():
	return 1.2 * (1.2 * (NB - GB) - 2.5 * (RB - GB))

#Modified Triangular Vegetation Index 2
def MTVI2():
	return (1.5 * (1.2 * (NB - GB) - 2.5 * (RB - GB))) / ((((2.0 * NB + 1) ** 2) - (6.0 * NB - 5 * (RB ** 0.5)) - 0.5) ** 0.5)

#Normalized Difference Drought Index
def NDDI():
	return (((NB - RB)/(NB + RB)) - ((GB - NB)/(GB + NB)))/(((NB - RB)/(NB + RB)) + ((GB - NB)/(GB + NB)))

#Normalized Difference Vegetation Index
def NDVI():
	return (NB - RB)/(NB + RB)

#Normalized Difference Yellowness Index
def NDYI():
	return (GB - BB) / (GB + BB)

#Normalized Green Red Difference Index
def NGRDI():
	return (GB - RB) / (GB + RB)

#Near-Infrared Reflectance of Vegetation
def NIRv():
	return ((NB - RB) / (NB + RB)) * NB

#Non-Linear Vegetation Index
def NLI():
	return ((NB ** 2) - RB)/((NB ** 2) + RB)

#Normalized Green
def NormG():
	return GB/(NB + GB + RB)

#Normalized NIR
def NormNIR():
	return NB/(NB + GB + RB)

#Normalized Red
def NormR():
	return RB/(NB + GB + RB)

#Optimized Soil-Adjusted Vegetation Index
def OSAVI():
	return (NB - RB) / (NB + RB + 0.16)

#Red Chromatic Coordinate
def RCC():
	return RB / (RB + GB + BB)

#Renormalized Difference Vegetation Index
def RDVI():
	return (NB - RB) / ((NB + RB) ** 0.5)

#Red Green Blue Vegetation Index
def RGBVI():
	return (GB ** 2.0 - BB * RB)/(GB ** 2.0 + BB * RB)

#Red-Green Ratio Index
def RGRI():
	return RB/GB

#Redness Index
def RI():
	return (RB - GB)/(RB + GB)

#Shadow Index
def SI():
	return ((1.0 - BB) * (1.0 - GB) * (1.0 - RB)) ** (1/3)

#Simple Ratio
def SR():
	return NB/RB

#Simple Ratio (800 and 550 nm)
def SR2():
	return NB/GB

#Transformed Difference Vegetation Index
def TDVI():
	return 1.5 * ((NB - RB)/((NB ** 2.0 + RB + 0.5) ** 0.5))

#Triangular Greenness Index
def TGI():
	return - 0.5 * (190 * (RB - GB) - 120 * (RB - BB))

#Transformed Vegetation Index
def TVI():
	return (((NB - RB)/(NB + RB)) + 0.5) ** 0.5

#Triangular Vegetation Index
def TriVI():
	return 0.5 * (120 * (NB - GB) - 200 * (RB - GB))

#Visible Atmospherically Resistant Index
def VARI():
	return (GB - RB) / (GB + RB - BB)

#Vegetation Index Green
def VIG():
	return (GB - RB) / (GB + RB)

#Blue Near-Infrared Reflectance of Vegetation
def bNIRv():
	return ((NB - BB)/(NB + BB)) * NB


