import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { CargaFileComponent } from './carga-file/carga-file.component';
import { ParametrosComponent } from './parametros/parametros.component';
import { ReporteComponent } from './reporte/reporte.component';

const routes: Routes = [
  {
    "path":"",
    component: CargaFileComponent
  },
  {
    "path":"parametrizacion",
    component: ParametrosComponent
  },
  {
    "path":"reporte",
    component: ReporteComponent
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
