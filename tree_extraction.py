
import numpy as np

def extract_trees(scan, params):


    M11 = 75
    M10 =  1 
    daa = 5*np.pi/306
    M2=1.5
    M2a = 10*np.pi/360 
    M3=3 
    M5=1 
    daMin2 = 2*np.pi/360 

    RR = scan

    AA = np.array(range(361))*np.pi/360

    ii1 = np.where(RR < M11)

    # ii1=Find16X(RR<M11) ;
    L1 = len(ii1)
    if L1<1:
        return []

    R1 = RR[ii1]
    A1 = AA[ii1]

    # ii2 = find(  (abs(diff(R1))>M2)|( diff(A1)>M2a) ) ;

    ii2 = np.flatnonzero((np.abs(np.diff(R1)) > M2) | (np.diff(A1) > M2a))

    L2= len(ii2)+1 
    ii2u = np.append(ii2, L1)
    ii2 = np.insert(ii2+1, 0, 0)
    # ii2u = int16([ ii2, L1 ])
    # ii2  = int16([1, ii2+1 ])

    # %ii2 , size(R1) ,

    R2  = R1[ii2 ]
    A2  = A1[ii2 ]

    A2u = A1[ii2u]
    R2u = R1[ii2u]

    x2  =  R2*np.cos(A2 )
    y2 =  R2*np.sin(A2 )
    x2u = R2u*np.cos(A2u)
    y2u = R2u*np.sin(A2u)

    flag=np.zeros(L2)

    L3=0 
    M3c= M3*M3
    
    if L2>1:
        L2m=L2-1
        dx2 = x2[1:L2]-x2u[:L2m] 
        dy2 = y2[1:L2]-y2u[:L2m]
        
        dl2 = dx2*dx2 + dy2*dy2
        ii3 = np.flatnonzero(dl2<M3c)
        L3=len(ii3)
        if L3>0:
            flag[ii3]=1
            flag[ii3+1]=1

        if L2>2:
            L2m = L2-2
            dx2 = x2[2:L2]-x2u[0:L2m]
            dy2 = y2[2:L2]-y2u[0:L2m]
            
            dl2 = dx2*dx2+dy2*dy2 
            ii3 = np.flatnonzero(dl2<M3c)
            L3b=len(ii3)
            if L3b>0:
                flag[ii3]=1
                flag[ii3+2]=1
                L3=L3+L3b

            if L2>3: 
                L2m=L2-3 
                dx2 = x2[3:L2]-x2u[0:L2m] 
                dy2 = y2[3:L2]-y2u[0:L2m]
                
                dl2 = dx2*dx2+dy2*dy2
                ii3 = np.flatnonzero(dl2<M3c)
                L3b=len(ii3)
                if L3b>0:
                     flag[ii3]=1
                     flag[ii3+3]=1
                     L3=L3+L3b


    if L2>1:
        ii3 = np.array(range(L2 - 1))
        ii3 = np.flatnonzero( (A2[ii3+1]-A2u[ii3])<daMin2 ) # objects close (in angle) from viewpoint.
        L3b=len(ii3) 
        if L3b>0:
            ff = (R2[ii3+1]>R2u[ii3])	# which object is in the back?
            ii3=ii3+ff	
            flag[ii3]=1			# mark them for the deletion
            L3=L3+L3b
        iixx=ii3

    if L3>0:
        ii3= np.flatnonzero(flag == 0)
        L3=len(ii3)
        ii4 = ii2[ii3].astype(np.float64)
        ii4u = ii2u[ii3].astype(np.float64) 
        R4  = R2[ii3]
        R4u  = R2u[ii3]  
        A4  = A2[ii3]
        A4u  = A2u[ii3]  
        x4  = x2[ii3]
        y4   = y2[ii3]   
        x4u = x2u[ii3]
        y4u  = y2u[ii3]  
    else:
        ii4 = ii2.astype(np.float64)
        ii4u = ii2u.astype(np.float64)
        R4  = R2
        R4u  = R2u
        A4  = A2
        A4u  = A2u
        x4  = x2
        y4   = y2 
        x4u = x2u
        y4u  = y2u 

    dx2 = x4-x4u
    dy2 = y4-y4u 
    dl2 = dx2*dx2+dy2*dy2 
    
    ii5 = np.flatnonzero(dl2<(M5*M5))
    L5 = len(ii5)
    if L5<1:
        return []

    R5 = R4[ii5]
    R5u = R4u[ii5]
    A5 = A4[ii5]
    A5u = A4u[ii5]
    ii4=ii4[ii5]
    ii4u=ii4u[ii5]

    ii5= np.flatnonzero( (R5>M10)&(A5>daa)&(A5u<(np.pi-daa)) )

    L5 = len(ii5)
    if L5<1:
        return []

    R5 = R5[ii5]
    R5u = R5u[ii5]
    A5 = A5[ii5]
    A5u = A5u[ii5]
    ii4=ii4[ii5]
    ii4u=ii4u[ii5]
    dL5 = (A5u+np.pi/360-A5)*(R5+R5u)/2 

    compa = ( np.abs(R5-R5u)<(dL5/3) ) 

    ii6 = np.flatnonzero(~compa)
    ii6=ii4[ii6] 

    ii5 = np.flatnonzero(compa)
    L5 = len(ii5)
    if L5<1:
        return []

    R5 = R5[ii5]
    R5u = R5u[ii5]
    A5 = A5[ii5]
    A5u = A5u[ii5]
    ii4=ii4[ii5]
    ii4u=ii4u[ii5]
    dL5 =dL5 [ii5]

    auxi = (ii4+ii4u)/2
    iia= np.floor(auxi)
    iib= np.ceil(auxi)

    Rs = (R1[iia.astype(int)]+R1[iib.astype(int)])/2

    ranges = Rs + dL5 / 2.0
    angles = (A5 + A5u) / 2.0 - np.pi/2
    diameters = dL5

    return list(zip(ranges, angles, diameters))