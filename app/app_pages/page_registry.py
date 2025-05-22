from app_pages.home.home import HomePage
from app_pages.util_pages.utilpage import UtilPage
from app_pages.manage.page import ManageWorkPage
from app_pages.factory_manage.page import FactoryManageWorkPage

PAGE_INSTANCES = {
    "render_home": HomePage(),  # YAMLの function: render_home に合わせる
    "render_util_page": UtilPage(),
    "manage_work_controller": ManageWorkPage(),
    "factory_manage_work_controller": FactoryManageWorkPage(),
}
