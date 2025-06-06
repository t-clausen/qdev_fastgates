import numpy as np
import qutip as qt
#import qutip_cupy
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers as pylab_helpers
def is_figure_active():
    return len(pylab_helpers.Gcf.figs) > 0
config = {}
try:
    with open("config.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            config[line.split(":")[0]] = eval(line.split(":")[1])
except FileNotFoundError:
    pass

dotplotinited = False
axs = None
fig = None
bloch = None
k = 0
def evaluate(results,ideal_gate,light=False):#!validate
    global dotplotinited, axs, fig, bloch, k
    """
    Do a qutip simulation, where the inverse of the ideal gate is used to run the simulation back. Then take the amplitude |initial> in the final state
    """
    

    #print(results)
    propabilities = []
    #expand the basis
    base_len = len(results[0].states[0].full())
    ideal_gate_inverse = np.eye(base_len,base_len,dtype=complex)
    ideal_gate_inverse[:2,:2] = np.array(ideal_gate).conj().T

    sx_,sy_,sz_ = qt.sigmax(),qt.sigmay(),qt.sigmaz()
    #sx, sy, sz = np.eye(base_len,dtype=complex), np.eye(base_len,dtype=complex), np.eye(base_len,dtype=complex)
    sx, sy, sz = np.zeros((base_len,base_len),dtype=complex), np.zeros((base_len,base_len),dtype=complex), np.zeros((base_len,base_len),dtype=complex)
    sx[:2,:2], sy[:2,:2], sz[:2,:2] = sx_.full(), sy_.full(), sz_.full()

    dotbloch = False
    if dotbloch and (not dotplotinited or not is_figure_active()):
        #fig,axs = plt.subplots(1,2,figsize=(6,4))
        fig = plt.figure(figsize=(6,2), dpi=600)
        axs = [fig.add_subplot(131),fig.add_subplot(132, projection='3d'),fig.add_subplot(133)]
        dummyax = fig.add_subplot(111, frame_on=False)
        dummyax.set_xticks([])
        dummyax.set_yticks([])
        #set ratios of the three widths
        #axs[0].set_aspect(2)
        #axs[1].set_aspect(2)
        #axs[2].set_aspect(2)
        dummyax.annotate(rf"$X_{{\pi/2}}$ by RWA pulse parameters, $t_g=1$", xy=(0.3, 1.2), ha='center', va='center', fontsize=12, xycoords='axes fraction')
        #dummyax.annotate(rf"$X_{{\pi/2}}$ by RWA pulse parameters, $t_g=1$", xy=(0.3, 1.15), ha='center', va='center', fontsize=12, xycoords='axes fraction')
        axs[2].set_title(rf"Infidelity vs. $t_g$", fontsize=12, y=1.07)
        bloch = qt.Bloch(axes=axs[1])
        dotplotinited = True
        k = 0

    for j,result in enumerate(results):
        #plot states on bloch sphere
        initial_state = result.states[0]
        final_state = result.states[-1]
        final_state = np.dot(np.dot(ideal_gate_inverse,final_state.full()),ideal_gate_inverse.T.conj())
        final_state = qt.Qobj(final_state)
        nextlast_state = result.states[-2]
        #print((final_state-result.states[-2]).full())
        if config["bloch"] and not light and not dotbloch:
            print("Plotting bloch sphere")
            ax=[0,0]
            plt.show()
            plt.clf()
            plt.close('all')
            if not is_figure_active():
                fig = plt.figure()
                ax[0] = fig.add_subplot(121, projection='3d')
                ax[1] = fig.add_subplot(122)
                bloch = qt.Bloch(fig=fig, axes=ax[0])
                exp_x = [np.trace(np.dot(sx,state.full())).real for state in result.states]
                exp_y = [np.trace(np.dot(sy,state.full())).real for state in result.states]
                exp_z = [np.trace(np.dot(sz,state.full())).real for state in result.states]
                bloch.add_points([exp_x,exp_y,exp_z],meth="l")
                ax[1].plot(range(len(result.states)), exp_x, label=r"$\sigma_x$")
                ax[1].plot(range(len(result.states)), exp_y, label=r"$\sigma_y$")
                ax[1].plot(range(len(result.states)), exp_z, label=r"$\sigma_z$")
                #plot final state as a red x on the bloch sphere
                fx = np.trace(np.dot(sx,final_state.full()))
                fy = np.trace(np.dot(sy,final_state.full()))
                fz = np.trace(np.dot(sz,final_state.full()))
                bloch.add_points([fx,fy,fz],meth="s",colors="red",alpha=0.5)
                bloch.add_points([exp_x[-1],exp_y[-1],exp_z[-1]],meth="s",colors="orange",alpha=0.5)
                nlx = np.trace(np.dot(sx,nextlast_state.full()))
                nly = np.trace(np.dot(sy,nextlast_state.full()))
                nlz = np.trace(np.dot(sz,nextlast_state.full()))
                bloch.add_points([nlx,nly,nlz],meth="s",colors="green",alpha=0.5)
                bloch.show()
                ax[1].set_xlabel("Timesteps")
                ax[1].set_ylabel("Expectation value")
                ax[1].legend()
                #bloch.legend()
                ax[0].set_xticks([0,150])#!tbd
                ax[0].set_xticklabels([r"$t_0$",r"$t_0+t_g$"])
            
                try:
                    #plt.savefig(f"temp/bloch_{result.solver}.png")
                    plt.savefig("bloch.png")
                    pass
                except:
                    print("Could not save figure")
                #plt.show()
                plt.clf()
                fig.clf()
                #fig.close()
                plt.close('all')

        if dotbloch and j%6-2 == 0:
            cs = [
                '#1b9e77',
                '#d95f02',
                '#7570b3'
            ]
            exp_x = [np.trace(np.dot(sx,state.full())) for state in result.states]
            exp_y = [np.trace(np.dot(sy,state.full())) for state in result.states]
            exp_z = [np.trace(np.dot(sz,state.full())) for state in result.states]
            bloch.add_points([exp_x,exp_y,exp_z],meth="l",alpha=0.3,colors=[(0,0,0)])
            lblx, lbly, lblz = r"$\sigma_x$", r"$\sigma_y$", r"$\sigma_z$"
            curln = axs[0].lines
            if len(curln) > 0:
                lblx, lbly, lblz = None, None, None
            axs[0].plot(range(len(result.states)), exp_x, label=lblx, color=cs[0], alpha=0.3)
            axs[0].plot(range(len(result.states)), exp_y, label=lbly, color=cs[1], alpha=0.3)
            axs[0].plot(range(len(result.states)), exp_z, label=lblz, color=cs[2], alpha=0.3)
            #plot final state as a red x on the bloch sphere
            ix = np.trace(np.dot(sx,initial_state.full()))
            iy = np.trace(np.dot(sy,initial_state.full()))
            iz = np.trace(np.dot(sz,initial_state.full()))
            fx = np.trace(np.dot(sx,final_state.full()))
            fy = np.trace(np.dot(sy,final_state.full()))
            fz = np.trace(np.dot(sz,final_state.full()))
            bloch.add_points([fx,fy,fz],meth="s",colors="red")
            bloch.add_points([ix,iy,iz],meth="s",colors="blue")
            bloch.add_points([exp_x[-1],exp_y[-1],exp_z[-1]],meth="s",colors="orange")
            nlx = np.trace(np.dot(sx,nextlast_state.full()))
            nly = np.trace(np.dot(sy,nextlast_state.full()))
            nlz = np.trace(np.dot(sz,nextlast_state.full()))
            bloch.add_points([nlx,nly,nlz],meth="s",colors="green")

            axs[0].set_xlabel("Timesteps")
            axs[0].set_ylabel("Expectation value")
            axs[0].legend(fontsize=8, frameon=False)
            #remove grid on bloch
            bloch.frame_alpha = 0
            bloch.frame_width = 1
            bloch.point_size = [20,20,20,20]
            bloch.font_size = 12
            bloch.xlabel = ["$|x\\rangle$",""]
            bloch.ylabel = ["$|y\\rangle$",""]
            bloch.zlabel = ["$|0\\rangle$","$|1\\rangle$"]
            bloch.zlpos = [1.2,-1.3]
            bloch.size = [100,100]
            bloch.point_marker = ["x", "x", "x", "x"]
            #plt.show()
            k+=1

            """def set_wh(ax,w,h):
                l = ax.figure.subplotpars.left
                r = ax.figure.subplotpars.right
                t = ax.figure.subplotpars.top
                b = ax.figure.subplotpars.bottom
                figw = float(w)/(r-l)
                figh = float(h)/(t-b)
                ax.figure.set_size_inches(figw, figh)
                return ax
            ax[0] = set_wh(axs[0],2,2)
            ax[1] = set_wh(axs[1],2,2)"""
            
            if k >= 10:
                bloch.show()
            #plt.savefig("bloch.png")
            
            
            #put labels on the right side of the plot
            
            

        
        #find which opperator the initial state is an eigenvector of
        inits = qt.Qobj(initial_state.full())
        truth = inits.isherm and np.isclose(inits.tr(),1) and np.isclose((inits*inits).tr(),1)
        if not truth:
            print("The initial state is not a pure state!")
        #print("Trace: ",final_state.tr())
        #print(final_state.full())
        eigvals,eigkets = initial_state.eigenstates()
        eigkets,eigvals = [eigket.full() for i,eigket in enumerate(eigkets) if np.isclose(eigvals[i],1)], [eigval for eigval in eigvals if np.isclose(eigval,1)] 
        initial_pure,eigval = eigkets[0],eigvals[0]#!incompatible eigvals
        #print(initial_pure)
        initial_type = None
        fs = final_state.full()
        A = np.dot(initial_pure,initial_pure.conj().T)
        #print(A)
        metric = np.dot(fs,A)
        fidelity = np.trace(metric)
        #fidelity = np.sum([initial_pure[i].conj()*initial_pure[i]*final_state.full()[i,i] for i in range(len(initial_pure))])
        #print("Pure_state: ",initial_pure[:2])
        #print("Fidelity: ",fidelity)
        #prob = np.trace(np.dot(initial_state.full().T.conj(),final_state.full()))#!why are expectation values complex?
        #!why are expectation values complex?
        propabilities.append(fidelity)
    fidelity = np.sum(propabilities)/len(propabilities)
    #bloch.show()
    #plt.show()
    return fidelity

def evaluate_onstate(final_states,initial_states,ideal_gate):#!validate
    """
    Do a qutip simulation, where the inverse of the ideal gate is used to run the simulation back. Then take the amplitude |initial> in the final state
    """
    

    #print(results)
    propabilities = []
    #expand the basis
    if hasattr(final_states[0],"states"):
        final_states = [state.states[-1] for state in final_states]
    base_len = len(final_states[0].full())
    ideal_gate_inverse = np.eye(base_len,base_len,dtype=complex)
    ideal_gate_inverse[:2,:2] = np.array(ideal_gate).conj().T

    sx_,sy_,sz_ = qt.sigmax(),qt.sigmay(),qt.sigmaz()
    #sx, sy, sz = np.eye(base_len,dtype=complex), np.eye(base_len,dtype=complex), np.eye(base_len,dtype=complex)
    sx, sy, sz = np.zeros((base_len,base_len),dtype=complex), np.zeros((base_len,base_len),dtype=complex), np.zeros((base_len,base_len),dtype=complex)
    sx[:2,:2], sy[:2,:2], sz[:2,:2] = sx_.full(), sy_.full(), sz_.full()



    for fstate, istate in zip(final_states,initial_states):
        #plot states on bloch sphere
        #initial_state = states.states[0]
        final_state = fstate
        final_state = np.dot(np.dot(ideal_gate_inverse,final_state.full()),ideal_gate_inverse.T.conj())
        final_state = qt.Qobj(final_state)
        #nextlast_state = sate.states[-2]
        #print((final_state-result.states[-2]).full())
        



        
        #find which opperator the initial state is an eigenvector of
        inits = qt.Qobj(istate.full())
        truth = inits.isherm and np.isclose(inits.tr(),1) and np.isclose((inits*inits).tr(),1)
        if not truth:
            print("The initial state is not a pure state!")
        #print("Trace: ",final_state.tr())
        #print(final_state.full())
        eigvals,eigkets = istate.eigenstates()
        eigkets,eigvals = [eigket.full() for i,eigket in enumerate(eigkets) if np.isclose(eigvals[i],1)], [eigval for eigval in eigvals if np.isclose(eigval,1)] 
        initial_pure,eigval = eigkets[0],eigvals[0]#!incompatible eigvals
        #print(initial_pure)
        initial_type = None
        fs = final_state.full()
        A = np.dot(initial_pure,initial_pure.conj().T)
        #print(A)
        metric = np.dot(fs,A)
        fidelity = np.trace(metric)
        #fidelity = np.sum([initial_pure[i].conj()*initial_pure[i]*final_state.full()[i,i] for i in range(len(initial_pure))])
        #print("Pure_state: ",initial_pure[:2])
        #print("Fidelity: ",fidelity)
        #prob = np.trace(np.dot(initial_state.full().T.conj(),final_state.full()))#!why are expectation values complex?
        #!why are expectation values complex?
        propabilities.append(fidelity)
    fidelity = np.sum(propabilities)/len(propabilities)
    return fidelity