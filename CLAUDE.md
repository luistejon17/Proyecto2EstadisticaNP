# Instrucciones para Claude — Proyecto 2 Estadística NP

## Remotes de git

Este repositorio tiene dos remotes:
- `origin` → GitHub (`https://github.com/luistejon17/Proyecto2EstadisticaNP.git`)
- `overleaf` → Overleaf vía integración git directa (`https://git.overleaf.com/6a0dda0c1f3b7ffd5be441af`). El token de autenticación NO está guardado aquí por seguridad — si el push falla por autenticación, pedir al usuario que lo proporcione.

La rama local es `main`. En Overleaf la rama se llama `master`. El comando de push a Overleaf es siempre:
```
git push overleaf main:master
```

## Reglas de comportamiento

### 1. Nunca hacer push sin autorización explícita
Nunca ejecutar `git push` (ni a `origin` ni a `overleaf`) a menos que el usuario lo pida explícitamente en ese mensaje. "Sube los cambios" o "push" cuentan como autorización. Hacer commits locales está permitido sin preguntar.

### 2. Flujo obligatorio para cambios en LaTeX
Cada vez que el usuario pida modificar archivos `.tex`, seguir este orden sin saltarse pasos:

1. **Hacer `git fetch overleaf`** y comparar con `HEAD` para detectar cambios hechos desde Overleaf.
2. Si hay cambios en Overleaf que no están en local: integrarlos con `git merge overleaf/master` (preferir la versión de Overleaf en conflictos de contenido del documento, salvo indicación contraria). Si hay conflictos no triviales, **avisar al usuario antes de hacer cualquier otra cosa**.
3. Una vez que local y Overleaf estén sincronizados, aplicar los cambios solicitados por el usuario.
4. Hacer commit local.
5. Hacer push a Overleaf (`git push overleaf main:master`). **No hacer push a GitHub** salvo que el usuario lo pida explícitamente.

## Contexto del proyecto

- Curso: MATE3526 / MATE4532 — Estadística No Paramétrica, 2026-10
- Entrega: miércoles 27 de mayo de 2026, 11 pm (Bloque Neón)
- Integrantes: Gabriel Sánchez, Juan David Duarte, Luis Tejón, Claudio (IA)
- Documento LaTeX en `documento/main.tex` (estructura modular con `\input`)
- Ver `proyecto2_2026 (actualizado).pdf` para el enunciado completo
