# Soluciones — Partes IV, V y VI

## Parte IV. Optimización con restricciones y programación lineal

### IV.A Multiplicadores de Lagrange

Se minimiza

\[
C(x,y)=x^2+2y^2
\]

sujeto a

\[
x+y=6.
\]

#### IV.A.1 Solución analítica

Con la convención indicada en el examen,

\[
\mathcal L(x,y,\lambda)=x^2+2y^2-\lambda(x+y-6).
\]

Las condiciones de estacionariedad son

\[
2x-\lambda=0,\qquad 4y-\lambda=0,\qquad x+y-6=0.
\]

De las dos primeras ecuaciones,

\[
\lambda=2x=4y \quad\Rightarrow\quad x=2y.
\]

Al sustituir en la restricción,

\[
2y+y=6 \Rightarrow y^*=2,\qquad x^*=4.
\]

Entonces

\[
\lambda^*=8
\]

y

\[
C^*=4^2+2(2^2)=24.
\]

#### IV.A.2 Globalidad y sensibilidad

La Hessiana de \(C\) es

\[
\nabla^2 C=
\begin{pmatrix}
2&0\\
0&4
\end{pmatrix},
\]

que es definida positiva. Por ello \(C\) es estrictamente convexa. La restricción \(x+y=6\) es afín, por lo que el punto factible estacionario es el mínimo global restringido y además es único.

Para la restricción general \(x+y=b\),

\[
x^*(b)=\frac{2b}{3},\qquad y^*(b)=\frac{b}{3}.
\]

El valor óptimo es

\[
V(b)=\frac{2}{3}b^2.
\]

Por tanto,

\[
V'(b)=\frac{4b}{3}.
\]

En \(b=6\),

\[
V'(6)=8=\lambda^*.
\]

Usando el multiplicador para estimar el cambio de \(b=6\) a \(b=6.1\),

\[
V(6.1)\approx V(6)+\lambda^*(0.1)=24.8.
\]

El valor exacto es \(V(6.1)=24.806666\ldots\), de modo que la aproximación lineal es consistente.

#### IV.A.3 Verificación

`src/parte_iv.py` resuelve el sistema con SymPy, verifica también la sustitución \(x=6-y\) y genera la figura de curvas de nivel, restricción y óptimo.

---

### IV.B Programación lineal

Se maximiza

\[
z=3x+2y
\]

sujeto a

\[
x+y\le 4,\qquad x\le 2,\qquad y\le 3,\qquad x\ge0,\qquad y\ge0.
\]

#### IV.B.1 Vértices

La enumeración de intersecciones factibles produce

\[
(0,0),\ (2,0),\ (2,2),\ (1,3),\ (0,3).
\]

#### IV.B.2 Evaluación del objetivo

\[
\begin{array}{c|c}
(x,y)&3x+2y\\
\hline
(0,0)&0\\
(2,0)&6\\
(2,2)&10\\
(1,3)&9\\
(0,3)&6
\end{array}
\]

El óptimo es

\[
(x^*,y^*)=(2,2),\qquad z^*=10.
\]

#### IV.B.3 Clasificación

Es un problema de **programación lineal (LP)** porque tanto la función objetivo como todas las restricciones son lineales.

En un poliedro factible no vacío y acotado, si una función lineal alcanza un óptimo, existe al menos un vértice óptimo. Aquí el máximo en \((2,2)\) es estrictamente mayor que en los demás vértices, por lo que es único.

---

## Parte V. Análisis numérico y regularización

### V.A Truncamiento, redondeo y orden de convergencia

Sea

\[
g(x)=\ln(1+x),\qquad g'(1)=\frac12.
\]

Se consideran

\[
D_h^+=\frac{g(1+h)-g(1)}{h}
\]

y

\[
D_h^0=\frac{g(1+h)-g(1-h)}{2h}.
\]

#### V.A.1 Taylor y orden teórico

Para la fórmula hacia adelante,

\[
g(1+h)=g(1)+hg'(1)+\frac{h^2}{2}g''(1)+O(h^3),
\]

por lo que

\[
D_h^+=g'(1)+\frac{h}{2}g''(1)+O(h^2).
\]

Como

\[
g''(1)=-\frac14,
\]

el término principal del error es

\[
D_h^+-g'(1)=-\frac{h}{8}+O(h^2).
\]

La fórmula es de **orden 1**.

Para la diferencia central,

\[
D_h^0=g'(1)+\frac{h^2}{6}g'''(1)+O(h^4).
\]

Como

\[
g'''(1)=\frac14,
\]

resulta

\[
D_h^0-g'(1)=\frac{h^2}{24}+O(h^4).
\]

La fórmula es de **orden 2**.

#### V.A.2 Régimen empírico

El script calcula \(h=10^{-k}\), \(k=1,\ldots,8\), usando `float64`.

En la diferencia hacia adelante se observa aproximadamente orden 1 durante buena parte de la tabla. En la diferencia central el orden cercano a 2 es muy claro hasta aproximadamente \(h=10^{-4}\). Después, el error de redondeo y la cancelación empiezan a dominar.

#### V.A.3 Por qué el error deja de mejorar

Reducir \(h\) disminuye el error de truncamiento, pero también obliga a restar números cada vez más próximos. En aritmética de precisión finita esa resta pierde dígitos significativos. Además, el error de redondeo queda dividido entre \(h\), por lo que puede amplificarse cuando \(h\) es demasiado pequeño.

La gráfica generada por el script muestra la transición entre el régimen dominado por truncamiento y el régimen dominado por precisión finita.

---

### V.B Condicionamiento y efecto de Ridge

Se estudia

\[
A_\varepsilon=
\begin{pmatrix}
1&1\\
1&1+\varepsilon
\end{pmatrix},
\qquad
\varepsilon=10^{-8},
\]

con

\[
b=
\begin{pmatrix}
2\\2+\varepsilon
\end{pmatrix},
\qquad
\delta b=
\begin{pmatrix}
0\\10^{-6}
\end{pmatrix}.
\]

#### V.B.1 Solución simbólica y número de condición

El sistema original tiene solución exacta

\[
x=
\begin{pmatrix}
1\\1
\end{pmatrix}.
\]

Si la segunda componente de \(b\) cambia en una cantidad \(\delta\),

\[
x_\delta=
\begin{pmatrix}
1-\delta/\varepsilon\\
1+\delta/\varepsilon
\end{pmatrix}.
\]

Con \(\delta=10^{-6}\) y \(\varepsilon=10^{-8}\), la solución cambia aproximadamente a

\[
(-99,101)^T.
\]

Los autovalores son

\[
\lambda_\pm=
\frac{2+\varepsilon\pm\sqrt{4+\varepsilon^2}}{2}.
\]

Para \(\varepsilon\) pequeño,

\[
\lambda_{\max}\approx2,\qquad
\lambda_{\min}\approx\frac{\varepsilon}{2},
\]

y por tanto

\[
\kappa_2(A_\varepsilon)\approx\frac{4}{\varepsilon}=4\times10^8.
\]

#### V.B.2 Amplificación

NumPy confirma un número de condición del orden de \(4\times10^8\). Una perturbación relativa muy pequeña en \(b\) produce un cambio relativo enorme en la solución. Esto es un efecto del mal condicionamiento del problema, no un fallo particular de `numpy.linalg.solve`.

#### V.B.3 Regularización

Se reemplaza \(A_\varepsilon\) por

\[
A_\varepsilon+\lambda I,\qquad \lambda=0.01.
\]

Los autovalores se desplazan en \(\lambda\), por lo que el menor deja de estar cerca de cero. El número de condición disminuye aproximadamente de \(4\times10^8\) a \(201\).

La solución regularizada es mucho menos sensible a la misma perturbación, aunque ya no coincide exactamente con la solución sin regularización. Esa diferencia es el sesgo introducido a cambio de estabilidad.

#### V.B.4 Relación con Ridge

En

\[
w_R=(X^TX+\lambda I)^{-1}X^Ty,
\]

el término \(\lambda I\) desplaza los autovalores de \(X^TX\) y evita divisiones por valores propios extremadamente pequeños o nulos. Esto mejora la estabilidad ante colinealidad o singularidad.

El compromiso es **sesgo–estabilidad**: una \(\lambda\) mayor reduce sensibilidad y magnitud de los coeficientes, pero también aleja la solución del ajuste no regularizado.

---

## Parte VI. Descenso batch y minibatch SGD

Los datos son

\[
x=(-1,0,1,2),\qquad y=(-2,0,2,4),
\]

con modelo

\[
\hat y_i=wx_i
\]

y pérdida

\[
L(w)=\frac14\sum_{i=1}^{4}(wx_i-y_i)^2.
\]

### VI.1 Solución analítica

Como \(y_i=2x_i\),

\[
wx_i-y_i=(w-2)x_i.
\]

Además,

\[
\sum x_i^2=1+0+1+4=6.
\]

Entonces

\[
L(w)=\frac{6}{4}(w-2)^2
=\frac32(w-2)^2
=\frac32w^2-6w+6.
\]

Por tanto,

\[
L'(w)=3w-6.
\]

El mínimo está en

\[
w^*=2,\qquad L(w^*)=0.
\]

### VI.2 Dos actualizaciones batch

Con \(w_0=0\) y \(\alpha=0.1\),

\[
L'(w_0)=-6
\]

y

\[
w_1=0-0.1(-6)=0.6.
\]

Luego,

\[
L(w_0)=6,\qquad L(w_1)=2.94.
\]

En \(w_1=0.6\),

\[
L'(w_1)=3(0.6)-6=-4.2,
\]

por lo que

\[
w_2=0.6-0.1(-4.2)=1.02
\]

y

\[
L(w_2)=1.4406.
\]

### VI.3 Implementación durante 30 épocas

El script implementa desde cero:

- GD batch: una actualización por época;
- minibatch SGD: tamaño 2, una permutación nueva por época y dos actualizaciones por época;
- semillas \(0,1,2,3,4\);
- pérdida completa registrada al final de cada época.

La tabla final y la gráfica semilogarítmica se guardan en `outputs/parte_vi/`.

### VI.4 Interpretación

**Gradiente insesgado.** Si el minibatch se elige uniformemente, cada observación tiene la probabilidad adecuada de contribuir al promedio. Por linealidad de la esperanza, el gradiente promedio del minibatch tiene como esperanza el gradiente completo.

**No monotonía.** Un gradiente de minibatch aproxima al gradiente completo y contiene variabilidad por muestreo. Por ello una actualización puede mejorar el minibatch usado pero aumentar temporalmente la pérdida sobre el conjunto completo. Reducir \(\alpha\) disminuye la magnitud de esas oscilaciones cerca del óptimo.

**Costo.** Comparar métodos solamente por época no demuestra superioridad universal. Batch usa todos los datos en una sola actualización; minibatch usa menos datos por actualización pero realiza más actualizaciones.

**Adam.** Adam combina un promedio móvil del gradiente (primer momento) con un promedio móvil del gradiente al cuadrado (segundo momento no centrado). El primero suaviza la dirección y el segundo adapta la escala efectiva de las actualizaciones.
