python2_script = f'''#!/usr/bin/env python
null=''
from VgPoint3 import *
from VgPoint2 import *
from VgMatrix import *
import VScn
import VGuiUtl
import VBrowserManager
import VExpMngr
import VCmdGui
import VCmd
import VCmdFramework
import VMaterial
import VMeshMdlr
import VToolKit
import VistaDb
NULL=VistaDb.PythonCNULL()
import VistaDb
#__________________ VhmCommand BEGIN __________________
var1=VCmd.Activate( 1, r"VHostManagerPlugin.VhmInterface", r"VhmCommand" )
import VHostManager
import VE
#__________________ SessionCommand BEGIN __________________
var2=VCmd.Activate( 1, r"VSessionManager.Command", r"SessionCommand" )
ret=VExpMngr.LoadFile( r"%s", 0 )
#__________________ ImportExportLocalProperties BEGIN __________________
var6=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var6, r"ImportFileName", r"%s" )
ret=VCmd.ExecuteCommand( var6, r"ImportLocalResults" )
#__________________ NewCompositeDisplay END __________________
VExpMngr.ExportFile( r"%s", 0 )'''

python2_script_X_vdbs = f'''#!/usr/bin/env python
null=''
from VgPoint3 import *
from VgPoint2 import *
from VgMatrix import *
import VScn
import VGuiUtl
import VBrowserManager
import VExpMngr
import VCmdGui
import VCmd
import VCmdFramework
import VMaterial
import VMeshMdlr
import VToolKit
import VistaDb
import VHostManager
import VE
NULL=VistaDb.PythonCNULL()
import VistaDb
var1=VCmd.Activate( 1, r"VHostManagerPlugin.VhmInterface", r"VhmCommand" )
var2=VCmd.Activate( 1, r"VSessionManager.Command", r"SessionCommand" )
ret=VExpMngr.LoadFile( r"%s", 0 )
ret=VHostManager.ChangeContext( r"Visual-RTM" )
var3=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"SimulationParameters" )
VCmd.SetStringValue( var3, r"MaterialDB", r"Model" )
VCmd.SetStringValue( var3, r"InjectedResin", r"RapeSeedOil" )
VCmd.SetDoubleValue( var3, r"MaxInjectionTime", %d.  )
%s
VCmd.Accept( var3 )
VCmd.Quit( var3 )
'''

import_lperm = r'''var%d=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var%d, r"ImportFileName", r"%s" )
ret=VCmd.ExecuteCommand( var%d, r"ImportLocalResults" )
'''

export_file = r'''VExpMngr.ExportFile( r"%s", 0 )
'''

export_unf_file = r'''var%d=VCmd.Activate(1, r"VRTMUtilities.VRTMInterface", r"SolverManager")
VCmd.SetStringValue( var%d, r"File Path", r"%s" )
ret=VCmd.ExecuteCommand( var%d, r"RunDataCAST" )
VCmd.Quit( var%d )
'''

python2_script_full = f'''#!/usr/bin/env python
null=''
from VgPoint3 import *
from VgPoint2 import *
from VgMatrix import *
import VScn
import VGuiUtl
import VBrowserManager
import VExpMngr
import VCmdGui
import VCmd
import VCmdFramework
import VMaterial
import VMeshMdlr
import VToolKit
import VistaDb
NULL=VistaDb.PythonCNULL()
import VistaDb
#__________________ VhmCommand BEGIN __________________
var1=VCmd.Activate( 1, r"VHostManagerPlugin.VhmInterface", r"VhmCommand" )
import VHostManager
import VE
#__________________ SessionCommand BEGIN __________________
var2=VCmd.Activate( 1, r"VSessionManager.Command", r"SessionCommand" )
#__________________ VEAction BEGIN __________________
var3=VCmd.Activate( 1, r"VToolKit.VSectionCutInterface", r"VEAction" )
ret=VE.ChangeContext( r"Visual-RTM" )
VE.SetActiveWindow( r"p1w1" )
ret=VExpMngr.LoadFile( r"%s", 0 )
VE.SetCurrentPage( 1 )
ret=VE.ModelChange( "M  @0" )
#__________________ NewCompositeDisplay BEGIN __________________
var4=VCmd.Activate( 1, r"VCompUtils.VCompUtilsCmdInterface", r"NewCompositeDisplay" )
VCmd.SetObjectValue( var4, r"CurrentModel", "M  @0" )
lst1_count,lst1 =  VScn.List( "  MAT 2 "  )
VCmd.SetObjectArrayValue( var4, r"SelectedMaterial", lst1_count, lst1 )
ret=VCmd.ExecuteCommand( var4, r"UpdateFringeDisplay" )
VCmd.SetIntValue( var4, r"ComponentType", 5 )
ret=VCmd.ExecuteCommand( var4, r"UpdateFringeDisplay" )
VCmd.Accept( var4 )
VCmd.Quit( var4 )
#__________________ NewCompositeDisplay END __________________
#__________________ ImportExportLocalProperties BEGIN __________________
var6=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var6, r"ImportFileName", r"%s" )
ret=VCmd.ExecuteCommand( var6, r"ImportLocalResults" )
#__________________ NewCompositeDisplay END __________________
VExpMngr.ExportFile( r"%s", 0 )'''