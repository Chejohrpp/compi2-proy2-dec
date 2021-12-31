import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { CargaFileComponent } from './carga-file/carga-file.component';
import { ParametrosComponent } from './parametros/parametros.component';
import { ReporteComponent } from './reporte/reporte.component';
import { NavegacionComponent } from './navegacion/navegacion.component';
import { MainService } from './servicios/main.service';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [
    AppComponent,
    CargaFileComponent,
    ParametrosComponent,
    ReporteComponent,
    NavegacionComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [MainService],
  bootstrap: [AppComponent]
})
export class AppModule { }
