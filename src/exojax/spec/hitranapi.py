"""API for HITRAN and HITEMP outside HAPI

"""

def extract_hitemp(parbz2,nurange,margin,tag):
    """extract .par between nurange[0] and nurange[-1]

    Args:
       parbz2: .par.bz2 HITRAN/HITEMP file (str)
       nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid 
       margin: margin for nurange (cm-1)
       tag: tag for directory and output file

    Return:
       path of output file (pathlib)

    """
    import sys, os, bz2, tqdm, pathlib
    infilepath=pathlib.Path(parbz2)
    outdir=infilepath.parent/pathlib.Path(tag)
    os.makedirs(str(outdir), exist_ok=True)
    outpath=outdir/pathlib.Path(infilepath.stem)
    
    numin=nurange[0]-margin
    numax=nurange[-1]+margin    
    alllines=bz2.BZ2File(str(infilepath), "r")
    
    f = open(str(outpath),'w')    
    for line in tqdm.tqdm(alllines,desc="Extract HITEMP"):
        nu = float(line[3:15])
        if nu <= numax and nu >= numin:
                if b'\r\n' in line[-2:]:  
                    f.write(line[:-2].decode("utf-8")+'\n' )
                else:  
                    f.write(line.decode("utf-8") )        
    alllines.close()
    f.close()
    return outpath

if __name__  == "__main__":
    nurange=[4200.0,4300.0]
    margin=1.0
    tag="ext"
    extract_hitemp("/home/kawahara/exojax/data/CH4/06_HITEMP2020.par.bz2",nurange,margin,tag)
