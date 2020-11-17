function val=getNumericalAttribute(attributes,attrName)

qq=attributes.getNamedItem(attrName);

if(isempty(qq)) val=[];return;end;
val=str2num(qq.getValue);

end