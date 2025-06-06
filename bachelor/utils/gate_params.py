from utils import param_object as po
import numpy as np
import sympy as sp
import scipy
import matplotlib.pyplot as plt

def compute_inv(args):
    t_g = args["t_g"]
    omega_01 = args["omega_01"]
    omega_12 = args["omega_12"]
    omega_23 = args["omega_23"]
    phi_opp = args["phi_opp"]
    phi_12 = phi_opp[1,2]
    #doB = args["doB"]
    N = args["N"]
    try:
        args['omega_d'] = args['omega_d'].subs(sp.Symbol('omega_{01}'), omega_01).subs(sp.Symbol('phi_{12}'), phi_12).subs(sp.Symbol('omega_{12}'), omega_12).subs(sp.Symbol('t_g'), t_g)
    except:
        pass
    #B_loc = B_matrix.subs(sp.Symbol('t_g'),t_g).subs(sp.Symbol('omega_{01}'), omega_01).subs(sp.Symbol('omega_{12}'), omega_12).subs(sp.Symbol('omega_{23}'), omega_23)
    matrix_belem00 = [
        [
            A_matrix_func(i,j,**args) + 2*B_matrix_func(i,j,**args) + A_matrix_func(j,i,**args) + 2*B_matrix_func(j,i,**args)
            for j in range(1, N+1)
        ] for i in range(1, N+1)
    ]
    matrix_belem01 = [
        [
            -1 for j in range(1, N+1)
        ]
    ]
    matrix_belem10 = [
        [1] for j in range(1, N+1)
    ]
    matrix_belem11 = [[0]]
    matrix = np.zeros((N+1,N+1), dtype=complex)
    matrix[0:N, 0:N] = matrix_belem00
    matrix[0:N,N] = matrix_belem01[0]
    matrix[N, 0:N] = np.array(matrix_belem10).flatten()
    matrix[N, N] = matrix_belem11[0][0]
    #print("Matrix:")
    #print(matrix)
    #invert
    try:
        matrix_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        matrix = np.array([
            [matrix[i][j]+np.random.uniform(-1e-10, 1e-10) for j in range(len(matrix[i]))] for i in range(len(matrix))]
        )
        matrix_inv = np.linalg.inv(matrix)
    c_arr = matrix_inv @ np.array([[0]*N+ [np.pi/2/t_g]]).flatten()
    c_arr = c_arr[:-1]
    #plot historgram
    plt.bar(range(len(c_arr)), c_arr)
    plt.title("c_arr")
    plt.xlabel("c_arr")
    plt.ylabel("count")
    plt.grid()
    plt.savefig("temp/c_arr_hist.png")
    plt.show()
    plt.close()
    return c_arr

def A_matrix_func(n, m,gn_func=None,gm_func=None,forbidden_intervals=None,omega_d=None, t_g=None, omega_01=None, omega_12=None, omega_23=None,N=None,doB=None,phi_opp=None):
    gn = gn_func.subs(sp.Symbol('n'), n)
    gn = gn.subs(sp.Symbol('t_g'), t_g)
    gm = gm_func.subs(sp.Symbol('m'), m)
    gm = gm.subs(sp.Symbol('t_g'), t_g)
    """try:
        [float(forbidden_intervals[i][0].evalf()) for i in range(len(forbidden_intervals))]
        float(t_g.evalf())
    except:
        raise ValueError("t_g or forbidden_intervals not evaluated to numerical values")"""
    fis = [
        [float(f.subs(sp.Symbol('t_g'), t_g).subs(sp.Symbol('omega_{01}'), omega_01).subs(sp.Symbol('omega_{12}'), omega_12).subs(sp.Symbol('omega_{23}'), omega_23).evalf()) if not type(f) == float else f for f in fi]
        for fi in forbidden_intervals
    ]
    infmask = np.array([np.all([np.isinf(f) for f in fi]) for fi in fis])
    fis = np.array(fis)[~infmask]
    #use scipy quad to evaluate the integral
    fs = sp.Symbol('f')
    gn_lfunc = sp.lambdify(fs, gn, modules=["numpy"])
    gm_lfunc = sp.lambdify(fs, gm, modules=["numpy"])
    def integrand(f):
        return gn_lfunc(f)*np.conjugate(gm_lfunc(f))
    import matplotlib.pyplot as plt
    X = np.linspace(-0.5,0.5,1000)
    Y = np.array([integrand(x) for x in X])
    if 1 < 0:
        plt.plot(X, Y)
        plt.title("Integrand")
        plt.xlabel("f")
        plt.ylabel("integrand")
        plt.grid()
        plt.savefig("integrand_tmp.png")
        plt.show()
        plt.close()
    if len(fis) == 0:
        return 0
    contributions = [
        scipy.integrate.quad(
            integrand,
            float(np.real(fis[i][0])),
            float(np.real(fis[i][1]))
        )[0] for i in range(len(fis))
    ]
    return sum(contributions)

def B_matrix_func(n, m,gn_func=None,gm_func=None,forbidden_intervals=None,omega_d=None, t_g=None, omega_01=None, omega_12=None, omega_23=None,N=None,doB=True,phi_opp=None):
    if not doB:
        return 0
    B_matrix = sp.Piecewise(
        (0.5*(sp.sin(omega_d*t_g)**2)/((1-(omega_d*t_g/(n*sp.pi))**2)*(1-(omega_d*t_g/(m*sp.pi))**2)), True),
        #(2*(sp.sin(omega_d*t_g)**2)+2*(sp.cos(omega_d*t_g)**2)/((n*sp.pi/(2*omega_d*t_g))**2-1), sp.Eq(sp.Mod(n, 2), 0) & sp.Eq(sp.Mod(m, 2), 1)),
        #(2*(sp.sin(omega_d*t_g)**2)+2*(sp.cos(omega_d*t_g)**2)/((m*sp.pi/(2*omega_d*t_g))**2-1), sp.Eq(sp.Mod(n, 2), 1) & sp.Eq(sp.Mod(m, 2), 0)),
        #(2+2*(sp.cos(omega_d*t_g)**2)*(1/((1-(2*omega_d*t_g/(n*sp.pi))**2)*(1-(2*omega_d*t_g/(m*sp.pi))**2))-1), sp.Eq(sp.Mod(n, 2), 1) & sp.Eq(sp.Mod(m, 2), 1))
    )
    return B_matrix

def get_gate_params(gate="demo"):
    if gate == "demo_x":

        #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = np.pi/4
        theta_b = np.pi*0
        t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = 0.22253833
        t_0 = sp.Symbol('t_0')
        t = sp.Symbol('t')
        t_g = sp.Symbol('t_g')
        known_t0 = False
        #envelope = lambda t: np.piecewise(t,[t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0], [0,0.5*(1-np.sin(np.pi*(t-t_0)/t_g)**2),0]) #!should this be t0 dependant?
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        #    [0,,0], [t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0])
        #!should this be t0 dependant?
        #polarization = [1,0]
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": True,
            "calibrate_Z": False,
            "name": "demo_x",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            "ideal": [[0,1j],[1j,0]],
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif gate == "RWA_x_nooptim":
        #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = 0
        theta_b = 0

        t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = sp.Symbol('omega_{01}')
        t_0 = sp.Symbol('t_0')
        t = sp.Symbol('t')
        t_g = sp.Symbol('t_g')
        #T1 = 30e3
        #T2 = 20e3
        envelope = sp.Piecewise(((sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "Omega_eq": np.pi/(t_g),
            "known_t0": False,
            "calibrate_Omega": False,
            "initial_Omega": np.pi/(t_g),
            "calibrate_Z": False,
            "name": "RWA_x",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            #"T1": T1,
            #"T2": T2,
        })
    elif gate == "RWA_x_DRAG_nooptim":
        #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = 0
        theta_b = 0

        t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = sp.Symbol('omega_{01}')
        t_0 = sp.Symbol('t_0')
        t = sp.Symbol('t')
        t_g = sp.Symbol('t_g')
        alpha = sp.Symbol('omega_{12}')-sp.Symbol('omega_{01}')#anharmonicity
        envelope_inphase = sp.Piecewise(((1-sp.cos(2*sp.pi*(t-t_0)/t_g)), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        envelope_quad = sp.Piecewise(((-2*sp.pi/(t_g*alpha)*sp.sin(2*sp.pi*(t-t_0)/t_g)), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        return po.GateParams({
            "omega_d": omega_d,
            "envelope_inphase": envelope_inphase,
            "envelope_quad": envelope_quad,
            "Omega_eq": np.pi/(t_g),
            "known_t0": False,
            "calibrate_Omega": False,
            "initial_Omega": np.pi/(t_g),
            "calibrate_Z": False,
            "name": "RWA_x_DRAG",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "c_ops": [
                #np.array([[0,158],[0,0]]),
                np.array([[0,158],[0,0]]),
                #np.array([[0,0],[0,0]]),
                #np.array([[0,1],[0,0]])
                np.array([[1,0],[0,-1]])
                #np.array([[0,0],[0,0]])
            ]
        })
    elif "corotating_xy" in gate:
        theta_a = np.pi/4
        theta_b = -np.pi/2
        omega_d = sp.Symbol('omega_{01}')#resonant
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        t = sp.Symbol('t')
        known_t0 = False
        calibrate_Omega = True
        Omega_eq = None
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        name = "corotating_xy"
        if "nooptim" in gate:
            calibrate_Omega = False
            Omega_eq= np.pi/t_g*np.sqrt(2)
            name = "corotating_xy_nooptim"
        
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": calibrate_Omega,
            "Omega_eq": Omega_eq,
            "initial_Omega": Omega_eq,
            "calibrate_Z": False,
            "name": name,
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
        })
    elif gate == "corotating_xy_virt_z":
        theta_a = np.pi/4
        theta_b = np.pi/2
        omega_d = sp.Symbol('omega_{01}')#resonant
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        t = sp.Symbol('t')
        known_t0 = False
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": True,
            "initial_Omega": np.pi/(t_g),
            "calibrate_VZ": True,
            "name": "corotating_xy",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
        })
    elif gate == "simple_x":
         #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = np.pi*0
        theta_b = np.pi*0
        #t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = sp.Symbol('omega_{01}')#resonant
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        t = sp.Symbol('t')
        known_t0 = False
        #envelope = lambda t: np.piecewise(t,[t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0], [0,0.5*(1-np.sin(np.pi*(t-t_0)/t_g)**2),0]) #!should this be t0 dependant?
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        #    [0,,0], [t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0])
        #!should this be t0 dependant?
        #polarization = [1,0]
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": True,
            "calibrate_Z": False,
            "initial_Omega": np.pi/(t_g),
            "name": "corotating_xy",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            #"ideal": [[1/np.sqrt(2),-1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]],#y gate
            "ideal": [
                [1/np.sqrt(2),-1j/np.sqrt(2)],
                [-1j/np.sqrt(2),1/np.sqrt(2)]
            ]
            #"ideal": [
            #    [1/np.sqrt(2),-0.5+0.5j],
            #    [0.5+0.5j,1/np.sqrt(2)]
            #]
        })
    elif "commensurate_x_virt_z" in gate:
        #Pi*(-a^2*omega^2 + Pi^2)*omega/(-2*a^3*omega^3 + 2*Pi^2*a*omega + 2*Pi^2*sin(omega*a))
        theta_a = 0
        theta_b = 0#from resonant pulse condition
        dt_0_init = -0.5*sp.Symbol('t_g')#!temp
        dt_0 = sp.Symbol('dt_0')
        t_g = sp.Symbol('t_g')
        #varphi = 0
        omega_d = sp.Symbol('omega_{01}')
        nt_0 = sp.Symbol('n')*np.pi/omega_d
        t_0 = nt_0+dt_0
        #tm = sp.Symbol('t_m')
        #t = tm+t_0+0.5*t_g
        t = sp.Symbol('t')
        tm = t-t_0-0.5*t_g
        calibrate_Omega = True
        calibrate_dt0 = True
        #Omega_eq= np.pi*(-t_g**2*omega_d**2 + np.pi**2)*omega_d/(-2*t_g**3*omega_d**3 + 2*np.pi**2*t_g*omega_d + 2*np.pi**2*sp.sin(omega_d*t_g))*4#maple formula
        Omega_eq = 3.25*(-4*np.pi**2*t_g**2*omega_d**3 + np.pi**4*omega_d)/((8*1j*omega_d**2*t_g**2*np.pi - 2*1j*np.pi**3 + 4*np.pi**2*t_g*omega_d)*sp.exp(2*1j*omega_d*(dt_0 + t_g)) + (-8*1j*omega_d**2*t_g**2*np.pi + 2*1j*np.pi**3 + 4*np.pi**2*t_g*omega_d)*sp.exp(2*1j*omega_d*dt_0) + 4*omega_d*t_g*(np.pi + 2)*(-2*t_g*omega_d + np.pi)*(2*t_g*omega_d + np.pi))#maple formula
        dt0_eq = -0.5*t_g
        dt0_eq = None
        name = "commensurate_x_virt_z"
        if "nooptim" in gate:
            calibrate_Omega = False
            calibrate_dt0 = True
            name += "_nooptim"

        envelope = sp.Piecewise((2*(sp.cos(sp.pi*tm/t_g)**2), (tm>=-0.5*t_g) & (tm<=0.5*t_g)), (0,True))

        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": True,
            "calibrate_Omega": calibrate_Omega,
            "calibrate_dt0": calibrate_dt0,
            "initial_dt0": dt_0_init,
            "initial_Omega": Omega_eq,
            "calibrate_VZ": True,
            
            "name": name,
            "Omega_eq": Omega_eq,
            "dt0_eq": dt0_eq,
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope": envelope,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "c_ops": [
                #np.array([[0,158],[0,0]]),
                np.array([[0,0],[0,0]]),
                #np.array([[0,1],[0,0]])
                np.array([[0,0],[0,0]])
            ]
        })
    elif "FAST-MAGNUS" in gate and "DRAG" not in gate:
        N=10
        c_arr = [sp.Symbol(f'c_{n}') for n in range(1, N+1)]
        t = sp.Symbol('t')
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        envelope = sp.Piecewise(
            (
                sum(
                    [c_arr[i-1] * (1-sp.cos(2*sp.pi*i*(t-t_0)/t_g)) for i in range(1, N+1)]
                ),
                (t>=t_0) & (t<=t_g+t_0))
            , (0,True))
        omega_01 = sp.Symbol('omega_{01}')
        omega_d = omega_01
        theta_a = 0
        theta_b = 0
        calibrate_Omega = True
        Omega_eq = None
        n,m = sp.symbols('n'), sp.symbols('m')
        n = sp.Symbol('n')
        m = sp.Symbol('m')
        
        F_x = sum(
            [c_arr[i-1]*c_arr[j-1] * (
                B_matrix_func(i, j, omega_d=omega_d, t_g=t_g)
            ) for i in range(1, N+1) for j in range(1, N+1)]
        )
        F_y = F_x
        cond_eq = sp.pi/4-t_g*sum(c_arr)
        omega_0 = 0
        omega_12 = sp.Symbol('omega_{12}')
        omega_02 = omega_01 + omega_12
        omega_03 = sp.Symbol('omega_{23}') + omega_12 + omega_01
        forbidden_intervals = [
            [(omega_02-omega_0)*0.99/(2*np.pi), (omega_02-omega_0)*1.01/(2*np.pi)],#the 0->2 transition
            #[(omega_12+omega_01)*0.5/(2*np.pi)-(omega_12-omega_01)/(2*np.pi),(omega_12+omega_01)*0.5/(2*np.pi)],# the 1->2 transition
            [0.99*omega_12/(2*np.pi),1.01*omega_12/(2*np.pi)],# the 1->2 transition
            [(omega_02+omega_01)/(2*np.pi),np.inf],# energy cutoff
        ]
        fs = sp.Symbol('f')


        gn_func = t_g*(
            sp.exp(-1j*sp.pi*t_g*fs)*sp.sinc(sp.pi*t_g*fs)
            -0.5*sp.exp(1j*sp.pi*(n/t_g-fs)*t_g)*sp.sinc(sp.pi*(n/t_g-fs)*t_g)
            -0.5*sp.exp(-1j*sp.pi*(n/t_g+fs)*t_g)*sp.sinc(sp.pi*(n/t_g+fs)*t_g)
        )
        gm_func = gn_func.subs(n, m)

            
        #Lagrangian = lambda: K_func() + F_x + F_y + cond_eq
        #Lagrangian = sp.simplify(Lagrangian)
        
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "get_matrix_inverse": True,
            "matrix_inverse": compute_inv,
            "calibrate_VZ": True,
            "gn_func": gn_func,
            "gm_func": gm_func,
            "doB": True,
            "name": gate,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope_inphase": envelope,
            "envelope_quad": sp.Piecewise((0, True)),#no quad envelope in this case
            "envelope_coffs": c_arr,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "forbidden_intervals": forbidden_intervals,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif "FAST-MAGNUS-DRAG" in gate:
        N=3
        c_arr = [sp.Symbol(f'c_{n}') for n in range(1, N+1)]
        t = sp.Symbol('t')
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        omega_01 = sp.Symbol('omega_{01}')
        alpha = sp.Symbol('omega_{12}')-omega_01#anharmonicity
        phi_12 = sp.Symbol('phi_{12}')#matrix element of the 1->2 transition
        envelope_inphase = sp.Piecewise(
            (
                sum(
                    [c_arr[i-1] * (1-sp.cos(2*sp.pi*i*(t-t_0)/t_g)) for i in range(1, N+1)]
                ),
                (t>=t_0) & (t<=t_g+t_0))
            , (0,True))
        envelope_quad = -0.5*sp.diff(envelope_inphase, t)*phi_12/alpha #the derivative of the in-phase envelope
        omega_d = omega_01-(3*(sp.pi**2))*(phi_12**2-2*phi_12)/(128*alpha*(t_g**2))#from the RWA condition
        theta_a = 0
        theta_b = 0
        calibrate_Omega = True
        Omega_eq = None
        n,m = sp.symbols('n'), sp.symbols('m')
        n = sp.Symbol('n')
        m = sp.Symbol('m')
        
        F_x = sum(
            [c_arr[i-1]*c_arr[j-1] * (
                B_matrix_func(i, j, omega_d=omega_d, t_g=t_g)
            ) for i in range(1, N+1) for j in range(1, N+1)]
        )
        F_y = F_x
        cond_eq = sp.pi/4-t_g*sum(c_arr)
        omega_0 = 0
        omega_12 = sp.Symbol('omega_{12}')
        omega_02 = omega_01 + omega_12
        omega_03 = sp.Symbol('omega_{23}') + omega_12 + omega_01
        forbidden_intervals = [
            [(omega_02-omega_0)*0.99/(2*np.pi), (omega_02-omega_0)*1.01/(2*np.pi)],#the 0->2 transition
            #[(omega_12+omega_01)*0.5/(2*np.pi)-(omega_12-omega_01)/(2*np.pi),(omega_12+omega_01)*0.5/(2*np.pi)],# the 1->2 transition
            [0.99*omega_12/(2*np.pi),1.01*omega_12/(2*np.pi)],# the 1->2 transition
            [(omega_02+omega_01)/(2*np.pi),np.inf],# energy cutoff
        ]
        fs = sp.Symbol('f')


        gn_func = t_g*(
            sp.exp(-1j*sp.pi*t_g*fs)*sp.sinc(sp.pi*t_g*fs)
            -0.5*sp.exp(1j*sp.pi*(n/t_g-fs)*t_g)*sp.sinc(sp.pi*(n/t_g-fs)*t_g)
            -0.5*sp.exp(-1j*sp.pi*(n/t_g+fs)*t_g)*sp.sinc(sp.pi*(n/t_g+fs)*t_g)
        )
        gm_func = gn_func.subs(n, m)

            
        #Lagrangian = lambda: K_func() + F_x + F_y + cond_eq
        #Lagrangian = sp.simplify(Lagrangian)
        
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "get_matrix_inverse": True,
            "matrix_inverse": compute_inv,
            "calibrate_VZ": True,
            "gn_func": gn_func,
            "gm_func": gm_func,
            "doB": True,
            "name": gate,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope_inphase": envelope_inphase,
            "envelope_quad": envelope_quad,
            "envelope_coffs": c_arr,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "forbidden_intervals": forbidden_intervals,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif gate == "FAST-DRAG":
        N=3
        c_arr = [sp.Symbol(f'c_{n}') for n in range(1, N+1)]
        t = sp.Symbol('t')
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        omega_01 = sp.Symbol('omega_{01}')
        alpha = sp.Symbol('omega_{12}')-omega_01#anharmonicity
        phi_12 = sp.Symbol('phi_{12}')#matrix element of the 1->2 transition
        envelope_inphase = sp.Piecewise(
            (
                sum(
                    [c_arr[i-1] * (1-sp.cos(2*sp.pi*i*(t-t_0)/t_g)) for i in range(1, N+1)]
                ),
                (t>=t_0) & (t<=t_g+t_0))
            , (0,True))
        #envelope_quad = -sp.diff(envelope_inphase, t)/alpha#the derivative of the in-phase envelope
        envelope_quad = -0.5*sp.diff(envelope_inphase, t)*phi_12/alpha #the derivative of the in-phase envelope
        omega_d = omega_01-(3*(sp.pi**2))*((phi_12**2)-2*phi_12)/(128*alpha*(t_g**2))
        theta_a = 0
        theta_b = 0
        calibrate_Omega = True
        Omega_eq = None
        n,m = sp.symbols('n'), sp.symbols('m')
        n = sp.Symbol('n')
        m = sp.Symbol('m')

        cond_eq = sp.pi/4-t_g*sum(c_arr)
        omega_0 = 0
        omega_12 = sp.Symbol('omega_{12}')
        omega_02 = omega_01 + omega_12
        omega_03 = sp.Symbol('omega_{23}') + omega_12 + omega_01
        forbidden_intervals = [
            [(omega_02-omega_0)*0.99/(2*np.pi), (omega_02-omega_0)*1.01/(2*np.pi)],#the 0->2 transition
            #[(omega_12+omega_01)*0.5/(2*np.pi)-(omega_12-omega_01)/(2*np.pi),(omega_12+omega_01)*0.5/(2*np.pi)],# the 1->2 transition
            [0.99*omega_12/(2*np.pi),1.01*omega_12/(2*np.pi)],# the 1->2 transition
            [(omega_02+omega_01)/(2*np.pi),np.inf],# energy cutoff
        ]
        fs = sp.Symbol('f')


        gn_func = t_g*(
            sp.exp(-1j*sp.pi*t_g*fs)*sp.sinc(sp.pi*t_g*fs)
            -0.5*sp.exp(1j*sp.pi*(n/t_g-fs)*t_g)*sp.sinc(sp.pi*(n/t_g-fs)*t_g)
            -0.5*sp.exp(-1j*sp.pi*(n/t_g+fs)*t_g)*sp.sinc(sp.pi*(n/t_g+fs)*t_g)
        )
        gm_func = gn_func.subs(n, m)

            
        #Lagrangian = lambda: K_func() + F_x + F_y + cond_eq
        #Lagrangian = sp.simplify(Lagrangian)
        
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "get_matrix_inverse": True,
            "matrix_inverse": compute_inv,
            "calibrate_VZ": True,
            "gn_func": gn_func,
            "gm_func": gm_func,
            "doB": False,
            "name": gate,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope_inphase": envelope_inphase,
            "envelope_quad": envelope_quad,
            "envelope_coffs": c_arr,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "forbidden_intervals": forbidden_intervals,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif gate == "FAST_nooptim":
        N=10
        c_arr = [sp.Symbol(f'c_{n}') for n in range(1, N+1)]
        t = sp.Symbol('t')
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        envelope = sp.Piecewise(
            (
                sum(
                    [c_arr[i-1] * (1-sp.cos(2*sp.pi*i*(t-t_0)/t_g)) for i in range(1, N+1)]
                ),
                (t>=t_0) & (t<=t_g+t_0))
            , (0,True))
        omega_01 = sp.Symbol('omega_{01}')
        omega_d = omega_01
        theta_a = 0
        theta_b = 0
        calibrate_Omega = True
        Omega_eq = None
        n,m = sp.symbols('n'), sp.symbols('m')
        n = sp.Symbol('n')
        m = sp.Symbol('m')

        cond_eq = sp.pi/4-t_g*sum(c_arr)
        omega_0 = 0
        omega_12 = sp.Symbol('omega_{12}')
        omega_02 = omega_01 + omega_12
        omega_03 = sp.Symbol('omega_{23}') + omega_12 + omega_01
        forbidden_intervals = [
            [(omega_02-omega_0)*0.99/(2*np.pi), (omega_02-omega_0)*1.01/(2*np.pi)],#the 0->2 transition
            #[(omega_12+omega_01)*0.5/(2*np.pi)-(omega_12-omega_01)/(2*np.pi),(omega_12+omega_01)*0.5/(2*np.pi)],# the 1->2 transition
            [0.99*omega_12/(2*np.pi),1.01*omega_12/(2*np.pi)],# the 1->2 transition
            [(omega_02+omega_01)/(2*np.pi),np.inf],# energy cutoff
        ]
        fs = sp.Symbol('f')


        gn_func = t_g*(
            sp.exp(-1j*sp.pi*t_g*fs)*sp.sinc(sp.pi*t_g*fs)
            -0.5*sp.exp(1j*sp.pi*(n/t_g-fs)*t_g)*sp.sinc(sp.pi*(n/t_g-fs)*t_g)
            -0.5*sp.exp(-1j*sp.pi*(n/t_g+fs)*t_g)*sp.sinc(sp.pi*(n/t_g+fs)*t_g)
        )
        gm_func = gn_func.subs(n, m)

            
        #Lagrangian = lambda: K_func() + F_x + F_y + cond_eq
        #Lagrangian = sp.simplify(Lagrangian)
        
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "get_matrix_inverse": True,
            "matrix_inverse": compute_inv,
            "calibrate_VZ": True,
            "gn_func": gn_func,
            "gm_func": gm_func,
            "doB": False,
            "name": gate,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope_inphase": envelope,
            "envelope_quad": sp.Piecewise((0, True)),#no quad envelope in this case
            "envelope_coffs": c_arr,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "forbidden_intervals": forbidden_intervals,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif "magnus1_x_virt_z" in gate:
        phi = 0
        omega_d = sp.Symbol('omega_{01}')
        theta_a = 0
        theta_b = 0
        Omega = sp.Symbol('Omega')
        t_g = sp.Symbol('t_g')
        t_0 = sp.Symbol('t_0')
        t_c = sp.pi/omega_d
        t = sp.Symbol('t')
        tm = t-t_0
        Omega_initial = np.pi/t_g/2
        lambda_ = sp.Symbol('lambda')
        Lambda = t_g/(2*np.pi/sp.Symbol('omega_{01}'))
        N_c = sp.ceiling(t_g/t_c)
        #lambda_initial = 2*np.pi*(N_c)/t_g
        lambda_initial = sp.Piecewise(
            (2*np.pi*(t_g/t_c)/t_g, Lambda < 1),
            (2*np.pi*(N_c)/t_g, Lambda >= 1)
        )
        envelope_inphase = sp.Piecewise((Omega*(1-sp.cos(2*sp.pi*tm/t_g)), (tm>=0*t_g) & (tm<=t_g)), (0,True))
        envelope_quad = sp.Piecewise((Omega*2*sp.pi/t_g/lambda_*sp.sin(2*sp.pi*tm/t_g), (tm>=0*t_g) & (tm<=t_g)), (0,True))
        name = "magnus1_x_virt_z"
        calibrate_Omega = True
        calibrate_lambda = True
        Omega_eq = None
        if "nooptim" in gate:
            calibrate_Omega = False
            calibrate_lambda = False
            Omega_eq= np.pi/t_g/2
            name += "_nooptim"
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "calibrate_Omega": calibrate_Omega,
            "initial_Omega": Omega_initial,
            "Omega_eq": Omega_eq,
            "calibrate_VZ": True,
            "calibrate_lambda": calibrate_lambda,
            "initial_lambda": lambda_initial,
            "lambda_eq": lambda_initial,
            "name": name,
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope_inphase": envelope_inphase,
            "envelope_quad": envelope_quad,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    
    else:
        raise ValueError("Gate not found")