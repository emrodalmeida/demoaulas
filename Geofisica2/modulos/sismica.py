import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

def refracao_sintetico(v1, v2, z, n_tracos, dx=1.0):
    # velocidades em m/s
    
    xx = (np.arange(n_tracos)) * dx

    t_direto = xx / v1
    t_refletido = np.sqrt(4*z**2 + xx**2)/v1
    
    if (v2 >= v1):
        t_refratado = (xx/v2) + (  (2*z*np.sqrt(v2**2 - v1**2)) / (v1*v2)  )
    else:
        t_refratado = np.ones(len(xx))*-1e-3

    it_crit = np.abs(t_refletido - t_refratado).argmin()

    fig = plt.figure(figsize=(15,7))
    ax = plt.subplot(1,1,1)
    ax.plot(xx, t_direto*1e3, '-k', label='Ondas diretas', linewidth=2)
    ax.plot(xx, t_refletido*1e3, '-b', label='Ondas refletidas', linewidth=2)
    ax.plot(xx[it_crit:], t_refratado[it_crit:]*1e3, '-r', label='Ondas refratadas', linewidth=2)
    ax.plot(xx[:it_crit+1], t_refratado[:it_crit+1]*1e3, '--r', label='Projeção das\nondas refratadas', linewidth=2)
    ax.legend(loc='lower right', fontsize='12')
    ax.grid(which='both')

    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(2))

    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(2))

    ax.set_title('Tempos de chegada sintéticos')
    ax.set_xlabel('Distância (m)', fontsize=14)
    ax.set_ylabel('Tempo (ms)', fontsize=14)
    ax.set_xlim(xx[0], xx[-1])
    ax.set_ylim(0, 1e3*np.max(t_refletido))
    plt.show()


def refracao_real(v1, v2, z, reflexao=False):
    xx = np.linspace(0, 9, 101)
    t_direto = xx / v1
    t_refletido = np.sqrt(4*z**2 + xx**2)/v1
    t_refratado = (xx/v2) + (  (2*z*np.sqrt(v2**2 - v1**2)) / (v1*v2)  )
    ti = (2*z*np.sqrt(v2**2 - v1**2)) / (v1*v2)

    fig = plt.figure(figsize=(20, 9))
    ax = plt.subplot(1,1,1)
    img = plt.imread('dados/sismograma.png')
    ax.imshow(img, extent=[0, 8.96, -0.10, 5.0])
    ax.plot(xx, t_direto, '-b', label='Ondas diretas', linewidth=3)
    ax.plot(xx, t_refratado, '-r', label='Ondas refratadas', linewidth=3)

    if reflexao:
        ax.plot(xx, t_refletido, '-y', label='Ondas refletidas', linewidth=3)

    ax.legend(loc='lower right', fontsize='12')

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.grid(which='both')
    ax.set_title('Tempos de chegada em dados reais')
    ax.set_xlabel('Distância (km)', fontsize=14)
    ax.set_ylabel('Tempo (s)', fontsize=14)
    ax.set_xlim([0, 7.4])
    ax.set_ylim([0, 4.0])

    plt.show()
    
    return v1, z
    
    
def t2x2(v1, z):
    
    xx = np.linspace(0, 9, 101)
    t_refletido = np.sqrt(4*z**2 + xx**2)/v1
    
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(xx, t_refletido, linewidth=2)
    ax1.set_xlabel('Distância (km)', fontsize=14)
    ax1.set_ylabel('Tempo (s)', fontsize=14)
    ax1.set_title('Tempo duplo de trânsito', fontsize=14)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax1.grid(which='both')
    ax1.set_xlim([0, 7.4])
    ax1.set_ylim([0, 4.0])
    

    ax2.plot(xx**2, (t_refletido)**2, linewidth=2)
    ax2.set_xlabel(r'Distância$^{2}$ (km$^{2}$)', fontsize=14)
    ax2.set_ylabel(r'Tempo$^{2}$ (s$^{2}$)', fontsize=14)
    ax2.set_title(r'Método $t^{2}x^{2}$', fontsize=14)

    ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    ax2.grid(which='both')
    ax2.set_xlim([0, 7.4**2])
    ax2.set_ylim([0, 4.0**2])
    

    plt.show()
    
    
def reflexao_sintetico(v1, v2, v3, t01, t02, t03, n_tracos=81, dx=1.0):
    xx = np.arange(n_tracos) * dx
    v_intervalar = np.array([v1, v2, v3])    # m/s
    t0 = np.array([t01, t02, t03])               # s

    # -------

    n_interfaces = len(v_intervalar)
    v_rms = np.zeros(n_interfaces)

    for i in range(1, n_interfaces+1):
        v_rms[i-1] = np.sqrt(np.sum((v_intervalar[:i]**2) * t0[:i]) / np.sum(t0[:i]))
        
    t_refletido = np.zeros((n_interfaces, len(xx)))
    t_direto = xx / v_rms[0]

    for n in range(n_interfaces):
        t_refletido[n, :] = np.sqrt( t0[n]**2 + (xx / v_rms[n])**2 )

    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    cores = ['k', 'b', 'r']
    labels = ['Reflexão na camada 1', 'Reflexão na camada 2', 'Reflexão na camada 3']

    for n in range(n_interfaces):
        ax1.plot(xx, 1e3*(t_refletido[n, :]), c=cores[n], label=labels[n], linewidth=2)

    ax1.legend(loc='best', fontsize=12)
    ax1.set_xlabel(r'Distância (m)', fontsize=14)
    ax1.set_ylabel(r'Tempo (ms)', fontsize=14)
    ax1.set_title('Tempo de trânsito', fontsize=14)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(2))
    ax1.set_xlim(0, xx[-1])
    ax1.set_ylim(0, 1e3*np.max(t_refletido))
    ax1.grid(which='both')

    for n in range(n_interfaces):
        ax2.plot(xx**2, (1e3*t_refletido[n, :])**2, c=cores[n], label=labels[n], linewidth=2)

    ax2.legend(loc='best', fontsize=12)
    ax2.set_xlabel(r'Distância$^{2}$ (m$^{2}$)', fontsize=14)
    ax2.set_ylabel(r'Tempo$^{2}$ (ms$^{2}$)', fontsize=14)
    ax2.set_title(r'Método $t^{2}x^{2}$', fontsize=14)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(500))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(200))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(50))
    ax2.set_xlim(0, xx[-1]**2)
    ax2.set_ylim(0, (1e3*np.max(t_refletido))**2)
    ax2.grid(which='both')
    
    
def reflexao_sismograma(v1, v2, v3, t01, t02, t03, n_tracos=81):
    xx = np.linspace(0, 3000, n_tracos)
    v_intervalar = np.array([v1, v2, v3])    # m/s
    t0 = np.array([t01, t02, t03])               # s
    
    n_interfaces = len(v_intervalar)
    v_rms = np.zeros(n_interfaces)

    for i in range(1, n_interfaces+1):
        v_rms[i-1] = np.sqrt(np.sum((v_intervalar[:i]**2) * t0[:i]) / np.sum(t0[:i]))

    t_refletido = np.zeros((n_interfaces, len(xx)))
    t_refratado = np.zeros((n_interfaces, len(xx)))
    t_direto = xx / v_rms[0]

    for n in range(n_interfaces):
        t_refletido[n, :] = np.sqrt( t0[n]**2 + (xx / v_rms[n])**2 )

    fig = plt.figure(figsize=(20,10))
    ax1 = plt.subplot(1,1,1)
    cores = ['blue', 'lightgreen', 'red']
    labels = ['Reflexão na camada 1', 'Reflexão na camada 2', 'Reflexão na camada 3']

    img = plt.imread('dados/sismograma_reflexao.png')
    ax1.imshow(img, extent=[0, 3000, 1.4, 0.0], aspect='auto')
    for n in range(n_interfaces):
        ax1.plot(xx, t_refletido[n, :], c=cores[n], label=labels[n], linewidth=3)

    ax1.legend(loc='best', fontsize=14)
    ax1.set_xlabel(r'Distância (m)', fontsize=14)
    ax1.set_ylabel(r'Tempo (s)', fontsize=14)
    ax1.set_title('Tempo de trânsito', fontsize=14)
    ax1.set_ylim(1.4, 0.0)

    fig = plt.figure(figsize=(20,10))
    ax2 = plt.subplot(1,2,1)
    ax3 = plt.subplot(1,2,2)
    
    for n in range(n_interfaces):
        ax2.plot(xx, t_refletido[n, :], c=cores[n], label=labels[n], linewidth=2)

    ax2.legend(loc='upper left', fontsize=12)
    ax2.set_xlabel(r'Distância (m)', fontsize=14)
    ax2.set_ylabel(r'Tempo (s)', fontsize=14)
    ax2.set_title('Tempo de trânsito', fontsize=14)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(500))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.10))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax2.set_xlim(xx[0], xx[-1])
    ax2.set_ylim(0, np.max(t_refletido))
    ax2.grid(which='both')
    ax2.set_ylim(1.4, 0.0)

    for n in range(n_interfaces):
        ax3.plot((xx**2)/1e6, (t_refletido[n, :])**2, c=cores[n], label=labels[n], linewidth=2)

    ax3.legend(loc='best', fontsize=12)
    ax3.set_xlabel(r'Distância$^{2}$ (m$^{2}$) x10$^{6}$', fontsize=14)
    ax3.set_ylabel(r'Tempo$^{2}$ (s$^{2}$)', fontsize=14)
    ax3.set_title(r'Método $t^{2}x^{2}$', fontsize=14)
    ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax3.set_xlim((xx[0]**2)/1e6, (xx[-1]**2)/1e6)
    ax3.set_ylim(0, (np.max(t_refletido))**2)
    ax3.grid(which='both')
    ax3.set_ylim(1.4**2, 0.0)

    plt.show()